"""Tests for compute.reproduce — WP4 c9.

Covers the pure logic (parse_git_ref, load_image_record,
emit_digest_match_assertion) and CLI smoke. The actual rebuild loop
(git clone + docker build) is exercised only opt-in with -m live
because it requires a Docker daemon.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import Dataset, Literal, URIRef
from rdflib.namespace import RDF, XSD
from typer.testing import CliRunner

from compute.reproduce import (
    ReproductionResult,
    app,
    emit_digest_match_assertion,
    load_image_record,
    parse_git_ref,
)
from ontology.prefixes import EARL, G_AUDIT, G_EVIDENCE, PROV, RTM


runner = CliRunner()


# ---------------------------------------------------------------------------
# parse_git_ref
# ---------------------------------------------------------------------------

def test_parse_git_ref_https_with_path():
    base, sha, path = parse_git_ref(
        "git+https://github.com/DynamicalSystemsGroup/ADCS-lifecycle-demo@abc123#compute/Dockerfile"
    )
    assert base == "https://github.com/DynamicalSystemsGroup/ADCS-lifecycle-demo"
    assert sha == "abc123"
    assert path == "compute/Dockerfile"


def test_parse_git_ref_local_fallback():
    base, sha, path = parse_git_ref(
        "git+local://unknown@uncommitted#compute/Dockerfile"
    )
    assert base is None
    assert sha == "uncommitted"
    assert path == "compute/Dockerfile"


def test_parse_git_ref_rejects_non_git_uri():
    with pytest.raises(ValueError, match="not a git"):
        parse_git_ref("https://example.com/foo@bar")


def test_parse_git_ref_rejects_missing_sha():
    with pytest.raises(ValueError, match="missing @"):
        parse_git_ref("git+https://github.com/Org/Repo")


# ---------------------------------------------------------------------------
# load_image_record
# ---------------------------------------------------------------------------

def _write_image_trig(tmp_path: Path, image_iri: str, digest: str, git_ref: str) -> Path:
    trig = tmp_path / "rtm.trig"
    ds = Dataset()
    g = ds.graph(URIRef(G_EVIDENCE))
    iri = URIRef(image_iri)
    g.add((iri, RDF.type, RTM.DockerImage))
    g.add((iri, RTM.contentHash, Literal(digest)))
    g.add((iri, RTM.gitRef, Literal(git_ref, datatype=XSD.anyURI)))
    ds.serialize(destination=str(trig), format="trig")
    return trig


def test_load_image_record_reads_digest_and_git_ref(tmp_path):
    trig = _write_image_trig(
        tmp_path,
        "urn:adcs:docker-image:sha256-abc",
        "sha256:abc",
        "git+https://github.com/Org/Repo@xyz#compute/Dockerfile",
    )
    iri, digest, git_ref = load_image_record(trig, "urn:adcs:docker-image:sha256-abc")
    assert str(iri) == "urn:adcs:docker-image:sha256-abc"
    assert digest == "sha256:abc"
    assert git_ref.startswith("git+")


def test_load_image_record_raises_when_image_missing(tmp_path):
    trig = _write_image_trig(
        tmp_path, "urn:adcs:docker-image:sha256-abc", "sha256:abc",
        "git+https://github.com/Org/Repo@xyz#compute/Dockerfile",
    )
    with pytest.raises(ValueError, match="no rtm:contentHash"):
        load_image_record(trig, "urn:adcs:docker-image:missing")


# ---------------------------------------------------------------------------
# emit_digest_match_assertion
# ---------------------------------------------------------------------------

def test_emit_digest_match_assertion_passed():
    ds = Dataset(default_union=True)
    result = ReproductionResult(
        image_iri=URIRef("urn:adcs:docker-image:sha256-abc"),
        recorded_digest="sha256:abc",
        git_ref="git+https://github.com/Org/Repo@xyz#compute/Dockerfile",
        rebuilt_digest="sha256:abc",
        matched=True,
        detail="rebuilt cleanly",
    )
    assertion = emit_digest_match_assertion(ds, result)
    g = ds.graph(URIRef(G_AUDIT))
    types = set(g.objects(assertion, RDF.type))
    assert RTM.DigestMatchAssertion in types
    assert EARL.Assertion in types
    assert PROV.Activity in types
    assert (assertion, EARL.outcome, EARL.passed) in g
    assert (assertion, EARL.mode, EARL.automatic) in g
    assert (assertion, EARL.subject, URIRef("urn:adcs:docker-image:sha256-abc")) in g


def test_emit_digest_match_assertion_failed():
    ds = Dataset(default_union=True)
    result = ReproductionResult(
        image_iri=URIRef("urn:adcs:docker-image:sha256-abc"),
        recorded_digest="sha256:abc",
        git_ref="git+https://github.com/Org/Repo@xyz#compute/Dockerfile",
        rebuilt_digest="sha256:different",
        matched=False,
        detail="digest mismatch",
    )
    assertion = emit_digest_match_assertion(ds, result)
    g = ds.graph(URIRef(G_AUDIT))
    assert (assertion, EARL.outcome, EARL.failed) in g


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

def test_reproduce_cli_help():
    import re
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Strip ANSI escape codes + collapse whitespace so terminal-width
    # wrapping in CI doesn't split a flag across a wrap boundary
    # (same hardening as tests/test_cli.py::_flatten_help).
    flat = re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", result.stdout))
    assert "--image-digest" in flat, f"missing in help:\n{result.stdout}"
    assert "--from-trig" in flat, f"missing in help:\n{result.stdout}"


def test_reproduce_cli_exits_2_when_input_missing(tmp_path):
    bogus = tmp_path / "nope.trig"
    result = runner.invoke(app, ["--image-digest", "sha256:abc", "--from-trig", str(bogus)])
    assert result.exit_code == 2


def test_reproduce_cli_exits_2_when_image_not_found(tmp_path):
    trig = _write_image_trig(
        tmp_path, "urn:adcs:docker-image:sha256-abc", "sha256:abc",
        "git+https://github.com/Org/Repo@xyz#compute/Dockerfile",
    )
    result = runner.invoke(
        app,
        ["--image-digest", "sha256:not-present", "--from-trig", str(trig)],
    )
    assert result.exit_code == 2
