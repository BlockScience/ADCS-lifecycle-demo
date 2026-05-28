"""Build the assembled rtm.ttl from rtm-edit.ttl + vendored imports.

This is the Python-only build path. It produces:

  - ontology/rtm.ttl              the assembled artifact (committed)
  - ontology/assembly_manifest.json   build provenance (committed)

Build steps:

  1. Load rtm-edit.ttl.
  2. Generate SysMLv2 equivalence axioms from sysml_term_map.csv into a
     separate "bindings" graph (the CSV is the single source of truth;
     the equivalence axioms in rtm-edit.ttl are checked against it).
  3. Validate that every upstream term rtm-edit.ttl references actually
     exists in the corresponding vendored ontology.
  4. Compute the manifest: artifact SHA-256, vendored-import hashes,
     per-import term counts, equivalence-axiom count.
  5. Re-serialize rtm-edit.ttl (with bindings merged in) as rtm.ttl with
     a build-time header. The vendored imports are NOT merged in by
     this script — that is the optional ROBOT path (build_ontology_robot.py
     or `make ontology-robot`). MIREOT extracts via ROBOT remain future
     work; this script proves the integration is well-formed.

Usage:
    uv run python -m scripts.build_ontology
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDFS

ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY_DIR = ROOT / "ontology"
IMPORTS_DIR = ONTOLOGY_DIR / "imports"

EDIT_FILE = ONTOLOGY_DIR / "rtm-edit.ttl"
OUT_FILE = ONTOLOGY_DIR / "rtm.ttl"
MANIFEST_FILE = ONTOLOGY_DIR / "assembly_manifest.json"
SYSML_MAP_FILE = ONTOLOGY_DIR / "sysml_term_map.csv"


SYSML_LOCAL_NS = "https://www.omg.org/spec/SysML/2.0/"
# The OMG SysMLv2 OWL rendering's namespace.
SYSML_OMG_NS = "http://www.omg.org/spec/SysML/20240501/"

# Parsimony gate (WP2 §4.C). rtm: is an integration ontology — it should
# contribute only convenience handles, hashing properties, and SHACL
# targets, never new epistemic vocabulary. The gate keeps that promise
# honest by failing the build if the assembled artifact grows past the
# budget. The budget is the current size (156 triples) + 200 headroom
# for WP3's rtm:DockerImage class + property set and other small adds.
# WP3 will bump this when it lands; bumping is a deliberate act, not a
# silent drift.
TRIPLE_BUDGET = 356
TRIPLE_BUDGET_RATIONALE = (
    "Integration ontology parsimony gate. Current size + 200 headroom. "
    "Bumped deliberately when a new term class lands (next: WP3 "
    "rtm:DockerImage). See WP2 subplan §4.C."
)


@dataclass(frozen=True)
class VendoredImport:
    name: str
    iri: str
    filename: str
    namespace: str  # the namespace prefix of terms we expect to find here


VENDORED: list[VendoredImport] = [
    VendoredImport("PROV-O",  "http://www.w3.org/ns/prov-o",         "prov-o.ttl",  "http://www.w3.org/ns/prov#"),
    VendoredImport("EARL",    "http://www.w3.org/ns/earl#",          "earl.ttl",    "http://www.w3.org/ns/earl#"),
    VendoredImport("OntoGSN", "https://w3id.org/OntoGSN/ontology",   "ontogsn.ttl", "https://w3id.org/OntoGSN/ontology#"),
    VendoredImport("P-PLAN",  "http://purl.org/net/p-plan",          "p-plan.ttl",  "http://purl.org/net/p-plan#"),
    VendoredImport("OSLC RM", "http://open-services.net/ns/rm#",     "oslc-rm.ttl", "http://open-services.net/ns/rm#"),
    VendoredImport("OSLC QM", "http://open-services.net/ns/qm#",     "oslc-qm.ttl", "http://open-services.net/ns/qm#"),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _referenced_terms(edit_graph: Graph, namespace: str) -> set[str]:
    """Return the local-name set of terms in `namespace` referenced by edit_graph.

    Excludes the namespace IRI itself (empty local name) — that comes from
    references like `<http://www.w3.org/ns/earl#>` in `owl:imports` lists.
    """
    referenced: set[str] = set()
    for s, p, o in edit_graph:
        for term in (s, p, o):
            if isinstance(term, URIRef) and str(term).startswith(namespace):
                local = str(term)[len(namespace):]
                if local:  # skip the namespace-self reference
                    referenced.add(local)
    return referenced


def _defined_terms(import_graph: Graph, namespace: str) -> set[str]:
    """Return the local-name set of terms in `namespace` defined in import_graph
    (anything that appears as a subject of a class- or property-declaration)."""
    defined: set[str] = set()
    for s in import_graph.subjects(unique=True):
        if isinstance(s, URIRef) and str(s).startswith(namespace):
            defined.add(str(s)[len(namespace):])
    return defined


def _load_sysml_term_map() -> list[dict[str, str]]:
    with SYSML_MAP_FILE.open() as f:
        return list(csv.DictReader(f))


def _verify_sysml_axioms(edit_graph: Graph, term_map: list[dict[str, str]]) -> list[str]:
    """Verify every term-map row has the corresponding equivalence axiom in
    edit_graph. Automated, fully specified — verification per the WP1 §4.4
    discipline (renamed from `_validate_sysml_axioms` in WP2 §4.D since this
    file is being touched anyway for the CSV column rename)."""
    errors: list[str] = []
    for row in term_map:
        local = URIRef(f"{SYSML_LOCAL_NS}{row['local_term']}")
        omg = URIRef(f"{SYSML_OMG_NS}{row['omg_iri']}")
        if row["kind"] == "Class":
            axiom = (local, OWL.equivalentClass, omg)
        elif row["kind"] == "Property":
            axiom = (local, OWL.equivalentProperty, omg)
        else:
            errors.append(f"sysml_term_map.csv: unknown kind {row['kind']!r} for {row['local_term']}")
            continue
        if axiom not in edit_graph:
            errors.append(f"Missing axiom: {row['local_term']} -> {row['omg_iri']} ({row['kind']})")
    return errors


def _validate_references(edit_graph: Graph) -> tuple[dict[str, dict], list[str]]:
    """For each vendored import, confirm every term rtm-edit references is defined.

    Returns (per-import-info, error-list).
    per-import-info has {name: {iri, filename, sha256, total_triples, referenced, missing}}.
    """
    info: dict[str, dict] = {}
    errors: list[str] = []
    for vi in VENDORED:
        path = IMPORTS_DIR / vi.filename
        if not path.exists():
            errors.append(f"Vendored import missing: {vi.filename}. Run `make fetch-imports`.")
            continue
        g = Graph()
        g.parse(path, format="turtle")
        referenced = _referenced_terms(edit_graph, vi.namespace)
        defined = _defined_terms(g, vi.namespace)
        missing = sorted(referenced - defined)
        # Filter out terms that are namespaces themselves (e.g., the ontology IRI)
        missing = [m for m in missing if m]
        if missing:
            errors.extend(
                f"{vi.name}: rtm-edit references {vi.namespace}{m} which is not defined in vendored {vi.filename}"
                for m in missing
            )
        info[vi.name] = {
            "iri": vi.iri,
            "namespace": vi.namespace,
            "filename": vi.filename,
            "sha256": _sha256(path),
            "total_triples": len(g),
            "referenced_terms": sorted(referenced),
            "referenced_count": len(referenced),
            "missing_terms": missing,
        }
    return info, errors


def _count_equivalence_axioms(g: Graph) -> int:
    return sum(1 for _ in g.triples((None, OWL.equivalentClass, None))) + sum(
        1 for _ in g.triples((None, OWL.equivalentProperty, None))
    )


def _count_subclass_axioms(g: Graph) -> int:
    return sum(1 for _ in g.triples((None, RDFS.subClassOf, None)))


def _count_subproperty_axioms(g: Graph) -> int:
    return sum(1 for _ in g.triples((None, RDFS.subPropertyOf, None)))


def build() -> int:
    print(f"Building ontology from {EDIT_FILE.relative_to(ROOT)} ...")

    edit_graph = Graph()
    edit_graph.parse(EDIT_FILE, format="turtle")
    print(f"  rtm-edit.ttl: {len(edit_graph)} triples")

    # Step 2: validate sysml_term_map against rtm-edit equivalence axioms
    term_map = _load_sysml_term_map()
    sysml_errors = _verify_sysml_axioms(edit_graph, term_map)
    if sysml_errors:
        for e in sysml_errors:
            print(f"  ERROR  {e}", file=sys.stderr)
        return 1
    print(f"  sysml_term_map.csv: {len(term_map)} rows, all present in rtm-edit.ttl")

    # Step 3: validate every referenced upstream term exists in its vendored import
    import_info, ref_errors = _validate_references(edit_graph)
    if ref_errors:
        for e in ref_errors:
            print(f"  ERROR  {e}", file=sys.stderr)
        return 1

    # Step 4: assemble rtm.ttl content
    #
    # Strategy: serialize edit_graph as Turtle with a regenerated header
    # noting this is a built artifact. The body content is identical to
    # rtm-edit.ttl minus its hand-written header.
    out_graph = Graph()
    out_graph += edit_graph
    # Bind prefixes for nicer output
    for prefix, ns in edit_graph.namespaces():
        out_graph.bind(prefix, ns)

    body_bytes = out_graph.serialize(format="turtle").encode("utf-8")

    build_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (
        f"# =============================================================================\n"
        f"# AUTO-GENERATED ARTIFACT — DO NOT EDIT DIRECTLY\n"
        f"#\n"
        f"# Built {build_time} by scripts/build_ontology.py from:\n"
        f"#   - ontology/rtm-edit.ttl\n"
        f"#   - ontology/sysml_term_map.csv (SysMLv2 equivalences)\n"
        f"#   - ontology/imports/*.ttl (validation only; full MIREOT merge is\n"
        f"#     the optional ROBOT path)\n"
        f"#\n"
        f"# To make changes: edit rtm-edit.ttl and re-run `make ontology`.\n"
        f"# =============================================================================\n\n"
    )
    final_bytes = header.encode("utf-8") + body_bytes

    OUT_FILE.write_bytes(final_bytes)
    artifact_sha = _sha256_bytes(final_bytes)
    print(f"  Wrote {OUT_FILE.relative_to(ROOT)} ({len(out_graph)} triples, sha256={artifact_sha[:12]}...)")

    # Step 4.5: triple-count budget gate (WP2 §4.C parsimony rule)
    total_triples = len(out_graph)
    if total_triples > TRIPLE_BUDGET:
        print(
            f"  ERROR  rtm.ttl exceeds triple budget: "
            f"{total_triples} > {TRIPLE_BUDGET}",
            file=sys.stderr,
        )
        print(f"         Rationale: {TRIPLE_BUDGET_RATIONALE}", file=sys.stderr)
        print(
            "         To raise the budget, bump TRIPLE_BUDGET in "
            "scripts/build_ontology.py and update the rationale comment.",
            file=sys.stderr,
        )
        return 1
    headroom = TRIPLE_BUDGET - total_triples
    print(f"  Parsimony: {total_triples}/{TRIPLE_BUDGET} triples ({headroom} headroom)")

    # Step 5: emit manifest
    manifest = {
        "build_time": build_time,
        "artifact": {
            "path": "ontology/rtm.ttl",
            "sha256": artifact_sha,
            "total_triples": total_triples,
            "subclass_axioms": _count_subclass_axioms(out_graph),
            "subproperty_axioms": _count_subproperty_axioms(out_graph),
            "equivalence_axioms": _count_equivalence_axioms(out_graph),
        },
        "triple_budget": {
            "value": TRIPLE_BUDGET,
            "rationale": TRIPLE_BUDGET_RATIONALE,
            "headroom": headroom,
        },
        "edit_source": {
            "path": "ontology/rtm-edit.ttl",
            "sha256": _sha256(EDIT_FILE),
        },
        "sysml_term_map": {
            "path": "ontology/sysml_term_map.csv",
            "sha256": _sha256(SYSML_MAP_FILE),
            "row_count": len(term_map),
        },
        "imports": import_info,
        # ADCS_ROBOT_VERIFIED is set by the Makefile's `ontology` target
        # after the ROBOT preflight + merge + reason + report step has
        # cleared. `make ontology-python` (no-Java path) leaves it unset,
        # so the manifest records `robot_used: false` and Stage 0 prints
        # the Python-only banner.
        "robot_used": os.environ.get("ADCS_ROBOT_VERIFIED", "0") == "1",
        "notes": (
            "Python assembly + ROBOT/ELK verification (canonical `make ontology` path)."
            if os.environ.get("ADCS_ROBOT_VERIFIED", "0") == "1"
            else "Python assembly only (`make ontology-python`; run `make ontology` "
                 "with Java + obo-robot installed for ROBOT/ELK verification)."
        ),
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"  Wrote {MANIFEST_FILE.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(build())
