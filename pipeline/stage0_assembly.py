"""Stage 0 — Ontology Assembly (the narrated first act).

Makes the otherwise-invisible build step visible in every pipeline run.
Reads ontology/assembly_manifest.json (produced by `make ontology` and
committed alongside rtm.ttl), verifies the artifact hash matches the
manifest, prints a narrative banner showing what was assembled, loads
the ontology into <rtm:ontology>, and emits a p-plan:Activity into
<adcs:plan-execution> recording that the stage ran.

The narrative banner is the data-driven version of the "what's in this
ontology" story — produced from the manifest, not hand-written, so it
always reflects reality.

Supports --rebuild to invoke `make ontology` before reading the manifest,
useful for live demos that want to show the build run end-to-end.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from rdflib import Dataset, Literal, URIRef
from rdflib.namespace import RDF, XSD

from ontology.prefixes import ADCS, G_PLAN_EXECUTION, P_PLAN, PROV, RTM
from pipeline.dataset import create_dataset, graph_for, load_into

ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY_DIR = ROOT / "ontology"
PIPELINE_DIR = ROOT / "pipeline"
MANIFEST_PATH = ONTOLOGY_DIR / "assembly_manifest.json"
ARTIFACT_PATH = ONTOLOGY_DIR / "rtm.ttl"
SHAPES_PATH = ONTOLOGY_DIR / "rtm_shapes.ttl"
INDIVIDUALS_PATH = ONTOLOGY_DIR / "rtm_individuals.ttl"
PLAN_PATH = PIPELINE_DIR / "plan.ttl"

# Stage-0 step IRI — referenced by the P-PLAN plan definition (pipeline/plan.ttl).
STAGE0_STEP_IRI = URIRef(f"{RTM}plan/step/OntologyAssembly")

BANNER_WIDTH = 78


class Stage0Error(RuntimeError):
    """Raised when Stage 0 cannot proceed (missing manifest, hash drift, etc.)."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise Stage0Error(
            f"Assembly manifest missing at {MANIFEST_PATH.relative_to(ROOT)}. "
            f"Run `make ontology` to produce it."
        )
    return json.loads(MANIFEST_PATH.read_text())


def _verify_artifact_hash(manifest: dict) -> None:
    if not ARTIFACT_PATH.exists():
        raise Stage0Error(
            f"Built artifact missing at {ARTIFACT_PATH.relative_to(ROOT)}. "
            f"Run `make ontology`."
        )
    expected = manifest["artifact"]["sha256"]
    actual = _sha256(ARTIFACT_PATH)
    if expected != actual:
        raise Stage0Error(
            f"Manifest/artifact hash mismatch:\n"
            f"  manifest:        {expected}\n"
            f"  current rtm.ttl: {actual}\n"
            f"Run `make ontology` to rebuild the manifest (or revert your edits)."
        )


def _print_banner(manifest: dict, *, ontology_triples: int, shape_count: int) -> None:
    rule = "─" * BANNER_WIDTH
    print()
    print("[Stage 0/8] Ontology Assembly")
    print(rule)

    build_time = manifest.get("build_time", "?")
    print(f"  Loading assembled rtm.ttl (built {build_time} from rtm-edit.ttl)")

    print("  Imports resolved:")
    for name in sorted(manifest.get("imports", {})):
        info = manifest["imports"][name]
        # Format example:
        #   ✓ PROV-O    1146 triples, 7 terms referenced
        triples = info["total_triples"]
        refs = info["referenced_count"]
        print(f"    OK  {name:<10} {triples:>5} triples,  {refs} terms referenced")

    art = manifest["artifact"]
    print(f"  SysMLv2 equivalence axioms: {art['equivalence_axioms']}")
    print(f"  Local rtm: integration glue: {art['subclass_axioms']} subclass + "
          f"{art['subproperty_axioms']} subproperty axioms")

    if manifest.get("robot_used"):
        print("  Verification: ROBOT merge + ELK reasoning + report PASS")
    else:
        print("  Verification: Python build (run `make ontology-robot` for ELK + report)")

    print(f"  Loaded into <rtm:ontology>: {ontology_triples} triples")
    print(f"  Closure-rule suite registered: {shape_count} SHACL shapes")
    print(rule)


def _count_shapes(ds: Dataset) -> int:
    """Count SHACL NodeShapes in <rtm:ontology>."""
    from ontology.prefixes import SH

    onto_g = graph_for(ds, "ontology")
    return sum(1 for _ in onto_g.subjects(RDF.type, SH.NodeShape, unique=True))


def _emit_plan_activity(ds: Dataset, started: datetime, ended: datetime) -> URIRef:
    """Record this Stage-0 execution as a p-plan:Activity in <adcs:plan-execution>."""
    from traceability.plan_execution import PIPELINE_AGENT

    plan_g = graph_for(ds, "plan_execution")
    activity_id = f"exec/Stage0-{started.strftime('%Y%m%dT%H%M%SZ')}"
    activity = ADCS[activity_id]
    plan_g.add((activity, RDF.type, P_PLAN.Activity))
    plan_g.add((activity, RDF.type, PROV.Activity))
    plan_g.add((activity, P_PLAN.correspondsToStep, STAGE0_STEP_IRI))
    plan_g.add((activity, PROV.startedAtTime,
                Literal(started.isoformat(), datatype=XSD.dateTime)))
    plan_g.add((activity, PROV.endedAtTime,
                Literal(ended.isoformat(), datatype=XSD.dateTime)))
    plan_g.add((activity, PROV.wasAssociatedWith, PIPELINE_AGENT))
    plan_g.add((PIPELINE_AGENT, RDF.type, PROV.SoftwareAgent))
    return activity


def _run_make_ontology() -> None:
    print("[Stage 0] --rebuild requested: running `make ontology` ...")
    proc = subprocess.run(
        ["make", "ontology"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # Echo make's output so the user sees the build narrative.
    if proc.stdout:
        for line in proc.stdout.splitlines():
            print(f"  | {line}")
    if proc.returncode != 0:
        if proc.stderr:
            for line in proc.stderr.splitlines():
                print(f"  | {line}", file=sys.stderr)
        raise Stage0Error(f"`make ontology` failed (rc={proc.returncode}).")


def run_stage_0(*, rebuild: bool = False) -> Dataset:
    """Execute Stage 0 and return a Dataset with the ontology loaded.

    Caller is responsible for adding the structural model (Stage 1) and
    any subsequent stage outputs.
    """
    if rebuild:
        _run_make_ontology()

    started = datetime.now(timezone.utc)
    manifest = _load_manifest()
    _verify_artifact_hash(manifest)

    ds = create_dataset()
    load_into(ds, "ontology", ARTIFACT_PATH)
    if INDIVIDUALS_PATH.exists():
        load_into(ds, "ontology", INDIVIDUALS_PATH)
    if SHAPES_PATH.exists():
        load_into(ds, "ontology", SHAPES_PATH)

    # Load the P-PLAN process model into <rtm:plan>.
    if PLAN_PATH.exists():
        load_into(ds, "plan", PLAN_PATH)

    onto_g = graph_for(ds, "ontology")
    ontology_triples = len(onto_g)
    shape_count = _count_shapes(ds)

    _print_banner(manifest, ontology_triples=ontology_triples, shape_count=shape_count)

    ended = datetime.now(timezone.utc)
    _emit_plan_activity(ds, started, ended)

    return ds
