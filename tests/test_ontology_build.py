"""Phase B — verify the build chain produces a self-consistent artifact.

These tests are runnable any time the committed rtm.ttl, manifest, and
vendored imports are in the repo (after `make ontology`). They do NOT
require ROBOT or a network connection — they read what is already on
disk and assert internal consistency.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rdflib import Graph

ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY_DIR = ROOT / "ontology"
IMPORTS_DIR = ONTOLOGY_DIR / "imports"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest() -> dict:
    return json.loads((ONTOLOGY_DIR / "assembly_manifest.json").read_text())


def test_manifest_artifact_hash_matches_rtm_ttl():
    manifest = _load_manifest()
    actual_sha = _sha256(ONTOLOGY_DIR / "rtm.ttl")
    assert manifest["artifact"]["sha256"] == actual_sha, (
        "Manifest is stale relative to ontology/rtm.ttl. "
        "Run `make ontology` to rebuild."
    )


def test_manifest_edit_source_hash_matches_rtm_edit_ttl():
    manifest = _load_manifest()
    actual_sha = _sha256(ONTOLOGY_DIR / "rtm-edit.ttl")
    assert manifest["edit_source"]["sha256"] == actual_sha, (
        "Manifest edit-source hash is stale. Run `make ontology` to rebuild."
    )


def test_manifest_records_artifact_triple_count():
    manifest = _load_manifest()
    g = Graph()
    g.parse(ONTOLOGY_DIR / "rtm.ttl", format="turtle")
    assert manifest["artifact"]["total_triples"] == len(g)


def test_vendored_imports_present_and_hashed():
    manifest = _load_manifest()
    for name, info in manifest["imports"].items():
        path = IMPORTS_DIR / info["filename"]
        assert path.exists(), f"Vendored import {info['filename']} missing"
        assert _sha256(path) == info["sha256"], (
            f"Vendored {info['filename']} content drifted from manifest. "
            f"Run `make fetch-imports && make ontology`."
        )


def test_every_referenced_term_resolves_in_a_vendored_import():
    """All terms in manifest's `referenced_terms` should be present in the
    vendored copy. The build script validates this; this test guards
    against the committed artifact drifting from the committed imports."""
    manifest = _load_manifest()
    for name, info in manifest["imports"].items():
        path = IMPORTS_DIR / info["filename"]
        g = Graph()
        g.parse(path, format="turtle")
        defined_iris = {str(s) for s in g.subjects(unique=True)}
        namespace = info["namespace"]  # term namespace (distinct from ontology IRI)
        missing = [
            local for local in info["referenced_terms"]
            if f"{namespace}{local}" not in defined_iris
        ]
        assert not missing, (
            f"{name}: referenced terms not defined in {info['filename']}: {missing}"
        )


def test_sysml_term_map_matches_equivalence_axioms_in_artifact():
    """Every row of sysml_term_map.csv should produce one equivalence
    axiom in rtm.ttl, in either owl:equivalentClass or owl:equivalentProperty."""
    import csv
    from rdflib.namespace import OWL

    g = Graph()
    g.parse(ONTOLOGY_DIR / "rtm.ttl", format="turtle")

    with (ONTOLOGY_DIR / "sysml_term_map.csv").open() as f:
        rows = list(csv.DictReader(f))

    equiv_props = {OWL.equivalentClass, OWL.equivalentProperty}
    found_pairs = {
        (str(s), str(o)) for s, p, o in g if p in equiv_props
    }

    sysml_ns = "https://www.omg.org/spec/SysML/2.0/"
    omg_ns = "http://www.omg.org/spec/SysML/20240501/"

    missing = []
    for row in rows:
        pair = (f"{sysml_ns}{row['local_term']}", f"{omg_ns}{row['omg_iri']}")
        if pair not in found_pairs:
            missing.append(row["local_term"])
    assert not missing, f"sysml_term_map.csv rows without matching axioms: {missing}"
