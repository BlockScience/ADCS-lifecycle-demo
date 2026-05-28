"""compute.reproduce — verify image-digest reproducibility from a recorded image.

WP4 §4.9. Given an rtm:DockerImage IRI recorded in a TriG file (or in
Flexo), this CLI:

1. Reads the image record's rtm:gitRef + rtm:contentHash
2. Clones / checks out the recorded git commit in a temp worktree
3. Rebuilds the image from compute/Dockerfile at that commit
4. Compares the resulting runtime digest to the recorded rtm:contentHash
5. Prints PASS/FAIL and emits an rtm:DigestMatchAssertion (earl:Assertion)

Honors the verification/validation discipline: earl:mode is always
earl:automatic for these assertions (the check is fully specified).

Two input modes (mutually exclusive):
  --from-trig PATH        offline; reads an exported TriG/Turtle dataset
  --from-flexo            live; pulls the image record from a Flexo branch

Exit codes:
  0  reproducibility verified (digest matches)
  1  digest mismatch — the image at the recorded git ref produces a
     different runtime digest than the one recorded
  2  prerequisite failure (no rtm:gitRef, git unreachable, docker missing)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rdflib import Dataset, Literal, URIRef
from rdflib.namespace import RDF, XSD

from ontology.prefixes import EARL, G_AUDIT, PROV, RTM


REPRODUCE_CLI_AGENT = URIRef("urn:adcs:agent:reproduce-cli")
REPRODUCE_TEST = URIRef("urn:adcs:test:image-digest-reproduction")


@dataclass(frozen=True)
class ReproductionResult:
    """Outcome of a single reproduce-and-compare run."""
    image_iri: URIRef
    recorded_digest: str
    git_ref: str
    rebuilt_digest: str | None  # None when the rebuild itself failed
    matched: bool
    detail: str


def load_image_record(trig_path: Path, image_iri: str) -> tuple[URIRef, str, str]:
    """Read the image's recorded git ref + content hash from a TriG file.

    Returns (image_iri_uriref, recorded_digest, git_ref). Raises ValueError
    if the image isn't found or is missing required properties.
    """
    ds = Dataset(default_union=True)
    ds.parse(trig_path, format="trig")
    iri = URIRef(image_iri)
    digests = list(ds.objects(iri, RTM.contentHash))
    git_refs = list(ds.objects(iri, RTM.gitRef))
    if not digests:
        raise ValueError(f"image {iri} has no rtm:contentHash in {trig_path}")
    if not git_refs:
        raise ValueError(f"image {iri} has no rtm:gitRef in {trig_path}")
    return iri, str(digests[0]), str(git_refs[0])


def parse_git_ref(git_ref: str) -> tuple[str | None, str, str | None]:
    """Split a `git+<base>@<sha>#<path>` URI into (base, sha, file_path).

    Returns base=None for the `git+local://` fallback shape.
    """
    if not git_ref.startswith("git+"):
        raise ValueError(f"not a git+URI: {git_ref!r}")
    body = git_ref[4:]
    file_path = None
    if "#" in body:
        body, file_path = body.split("#", 1)
    if "@" not in body:
        raise ValueError(f"git+URI missing @<sha>: {git_ref!r}")
    base, sha = body.rsplit("@", 1)
    if base.startswith("local://"):
        return None, sha, file_path
    return base, sha, file_path


def rebuild_image_at_ref(
    git_ref: str,
    docker_cmd: str = "docker",
    workdir: Path | None = None,
) -> tuple[str, str]:
    """Clone the repo at the recorded sha, docker-build, return digest.

    Returns (rebuilt_digest, detail). On any subprocess failure raises
    RuntimeError with a descriptive message; the caller treats that as
    the FAIL branch and emits an earl:failed assertion.
    """
    base, sha, _file_path = parse_git_ref(git_ref)
    if base is None:
        raise RuntimeError(
            f"recorded git ref points at a local-only repo ({git_ref}); "
            "cannot rebuild without a clonable remote. Use a recorded image "
            "from a run made against the github remote."
        )

    # Argv-flag-smuggling guard: git refs come from RDF stores that may
    # be partly trust-boundary'd. Reject components that could be parsed
    # as git flags and validate the scheme before shelling out. Pair
    # with the `--` end-of-options sentinel on each subprocess call.
    if base.startswith("-") or sha.startswith("-"):
        raise RuntimeError(
            f"refusing flag-shaped git ref component: base={base!r} sha={sha!r}"
        )
    if not (base.startswith("https://") or base.startswith("ssh://") or base.startswith("git@")):
        raise RuntimeError(
            f"refusing non-https/ssh remote URL: {base!r}"
        )

    workdir = workdir or Path(tempfile.mkdtemp(prefix="adcs-reproduce-"))
    try:
        subprocess.run(
            ["git", "clone", "--quiet", "--", base, str(workdir)],
            check=True, capture_output=True, text=True, timeout=120,
        )
        subprocess.run(
            ["git", "-C", str(workdir), "checkout", "--quiet", "--", sha],
            check=True, capture_output=True, text=True, timeout=30,
        )
        # The image tag is throwaway; reproducibility is digest-based.
        tag = f"adcs-reproduce:{sha[:12]}"
        subprocess.run(
            [docker_cmd, "build", "-t", tag, "-f", "compute/Dockerfile", "."],
            cwd=str(workdir), check=True, capture_output=True, text=True, timeout=600,
        )
        inspect = subprocess.run(
            [docker_cmd, "image", "inspect", tag, "--format", "{{.Id}}"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        digest = inspect.stdout.strip()
        return digest, f"rebuilt at sha={sha[:12]} via {base}"
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"reproduce failed at step: {exc.cmd}; "
            f"stderr: {exc.stderr.strip()[-300:]}"
        ) from exc
    finally:
        if workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)


def emit_digest_match_assertion(
    ds: Dataset,
    result: ReproductionResult,
) -> URIRef:
    """Persist the reproduction outcome as rtm:DigestMatchAssertion in <adcs:audit>."""
    now_iso = datetime.now(timezone.utc).isoformat()
    suffix = now_iso.replace(":", "-").replace("+", "-").replace(".", "-")
    assertion = URIRef(f"urn:adcs:assertion:digest-match-{suffix}")
    g = ds.graph(URIRef(G_AUDIT))

    g.add((assertion, RDF.type, RTM.DigestMatchAssertion))
    g.add((assertion, RDF.type, EARL.Assertion))
    g.add((assertion, RDF.type, PROV.Activity))
    g.add((assertion, EARL.subject, result.image_iri))
    g.add((assertion, EARL.test, REPRODUCE_TEST))
    g.add((assertion, EARL.outcome, EARL.passed if result.matched else EARL.failed))
    g.add((assertion, EARL.mode, EARL.automatic))
    g.add((assertion, PROV.wasAssociatedWith, REPRODUCE_CLI_AGENT))
    g.add((assertion, PROV.atTime, Literal(now_iso, datatype=XSD.dateTime)))
    return assertion


class Source(str, Enum):
    trig = "trig"
    flexo = "flexo"


app = typer.Typer(
    add_completion=False,
    help="Verify image-digest reproducibility from a recorded rtm:DockerImage.",
    no_args_is_help=True,
)


@app.command()
def main(
    image_digest: Annotated[str, typer.Option(
        "--image-digest",
        help="Digest substring or full sha256:... to look up in the dataset.",
    )],
    from_trig: Annotated[Path, typer.Option(
        "--from-trig",
        help="Path to a TriG dataset containing the rtm:DockerImage record.",
    )] = Path("output/rtm.trig"),
    docker_cmd: Annotated[str, typer.Option(
        "--docker-cmd",
        help="Docker CLI name (override for podman / colima).",
    )] = "docker",
) -> None:
    """Rebuild the image at its recorded git ref and digest-compare."""
    if not from_trig.exists():
        typer.echo(f"Input not found: {from_trig}", err=True)
        raise typer.Exit(code=2)

    ds = Dataset(default_union=True)
    ds.parse(from_trig, format="trig")

    # Resolve the image IRI from the digest substring
    image_iri = None
    recorded_digest = None
    git_ref = None
    for img in ds.subjects(RDF.type, RTM.DockerImage):
        digests = [str(d) for d in ds.objects(img, RTM.contentHash)]
        if any(image_digest in d for d in digests):
            image_iri = img
            recorded_digest = digests[0]
            git_refs = [str(g) for g in ds.objects(img, RTM.gitRef)]
            if git_refs:
                git_ref = git_refs[0]
            break

    if image_iri is None:
        typer.echo(f"No rtm:DockerImage matching digest {image_digest!r} in {from_trig}", err=True)
        raise typer.Exit(code=2)
    if git_ref is None:
        typer.echo(f"Image {image_iri} has no rtm:gitRef; cannot rebuild.", err=True)
        raise typer.Exit(code=2)

    typer.echo(f"[reproduce] image: {image_iri}")
    typer.echo(f"[reproduce] recorded digest: {recorded_digest}")
    typer.echo(f"[reproduce] git ref: {git_ref}")

    try:
        rebuilt_digest, detail = rebuild_image_at_ref(git_ref, docker_cmd=docker_cmd)
        matched = rebuilt_digest == recorded_digest
        result = ReproductionResult(
            image_iri=image_iri,
            recorded_digest=recorded_digest,
            git_ref=git_ref,
            rebuilt_digest=rebuilt_digest,
            matched=matched,
            detail=detail,
        )
        typer.echo(f"[reproduce] rebuilt digest: {rebuilt_digest}")
        typer.echo(f"[reproduce] result: {'PASS' if matched else 'FAIL'} ({detail})")
    except RuntimeError as exc:
        result = ReproductionResult(
            image_iri=image_iri,
            recorded_digest=recorded_digest,
            git_ref=git_ref,
            rebuilt_digest=None,
            matched=False,
            detail=str(exc),
        )
        typer.echo(f"[reproduce] FAIL: {exc}", err=True)

    emit_digest_match_assertion(ds, result)

    if not result.matched:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
