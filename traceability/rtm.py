"""Requirements Traceability Matrix assembly and validation.

The RTM is held in an rdflib Dataset with named graphs sized to match
Flexo MMS conventions (see pipeline/dataset.py and the named-graph IRIs
exported by ontology/prefixes). Existing SPARQL queries continue to
work unmodified because the Dataset is built with default_union=True.

Layer assignments:
- <rtm:ontology>      ontology/*.ttl (TBox + shapes + individuals)
- <adcs:structural>   structural/*.ttl (SysMLv2 instance data)
- <adcs:evidence>     evidence artifacts (populated by stage 4)
- <adcs:attestations> attestation events (populated by stage 6)
"""

from __future__ import annotations

from pathlib import Path

from rdflib import Dataset, Graph

from ontology.prefixes import ADCS, PROV, RTM, SYSML, bind_prefixes
from pipeline.dataset import create_dataset, graph_for, load_into
from traceability.queries import (
    ADCS_REQUIREMENTS,
    ATTESTATION_STATUS,
    REQUIREMENT_OUTCOMES,
    UNATTESTED_REQUIREMENTS,
    query_to_dicts,
)

ONTOLOGY_DIR = Path(__file__).resolve().parent.parent / "ontology"
STRUCTURAL_DIR = Path(__file__).resolve().parent.parent / "structural"

# Runtime TBox files (loaded into <rtm:ontology>). Explicit allow-list
# rather than glob — keeps ROBOT validation artifacts and rtm-edit.ttl
# (the build-source) out of the runtime graph.
_RUNTIME_ONTOLOGY_FILES = (
    "rtm.ttl",
    "rtm_individuals.ttl",
    "rtm_shapes.ttl",
)


def load_base_dataset() -> Dataset:
    """Build the base Dataset with ontology in <rtm:ontology> and
    structural model in <adcs:structural>.

    Only the canonical runtime TBox (rtm.ttl + rtm_individuals.ttl +
    rtm_shapes.ttl) is loaded. ROBOT validation artifacts, the
    hand-edited source rtm-edit.ttl, and the vendored imports under
    ontology/imports/ are build-time only and not part of the runtime.
    """
    ds = create_dataset()
    for filename in _RUNTIME_ONTOLOGY_FILES:
        load_into(ds, "ontology", ONTOLOGY_DIR / filename)
    for ttl in sorted(STRUCTURAL_DIR.glob("*.ttl")):
        load_into(ds, "structural", ttl)
    return ds


def load_base_graph() -> Dataset:
    """Backward-compatible alias. Returns the Dataset; consumers treating
    it as a Graph see the union view via default_union=True."""
    return load_base_dataset()


def assemble_rtm(
    base_graph: Graph | Dataset,
    evidence_graph: Graph,
) -> Graph | Dataset:
    """Merge evidence into the base.

    Two modes:
    - If base is a Dataset (the runtime), copy evidence triples into the
      <adcs:evidence> named graph and return the same Dataset.
    - If base is a plain Graph (legacy / test fixtures), produce a merged
      Graph that contains both — preserves the original signature.
    """
    if isinstance(base_graph, Dataset):
        ev_named = graph_for(base_graph, "evidence")
        for triple in evidence_graph:
            ev_named.add(triple)
        return base_graph

    # Legacy path: flat Graph merge
    rtm = Graph()
    bind_prefixes(rtm)
    for triple in base_graph:
        rtm.add(triple)
    for triple in evidence_graph:
        rtm.add(triple)
    return rtm


def verify_structural_completeness(graph: Graph) -> list[str]:
    """Verify that every ADCS requirement has at least one satisfy link.

    Returns a list of issues (empty if all good). Automated, fully
    specified — verification per the WP1 §4.4 discipline.
    """
    issues = []
    reqs = query_to_dicts(graph, ADCS_REQUIREMENTS)

    for req in reqs:
        name = req["name"]
        # Check for satisfy links
        q = f"""
        SELECT (COUNT(?rel) AS ?cnt) WHERE {{
            ?req sysml:declaredName "{name}" ;
                 sysml:ownedRelationship ?rel .
            ?rel a sysml:SatisfyRequirementUsage .
        }}
        """
        rows = query_to_dicts(graph, q)
        if rows and int(rows[0]["cnt"]) == 0:
            issues.append(f"{name}: no SatisfyRequirementUsage found")

    return issues


def verify_evidence_completeness(graph: Graph) -> list[str]:
    """Verify that every ADCS requirement has associated evidence.

    Returns a list of issues (empty if all good). Automated, fully
    specified — verification per the WP1 §4.4 discipline.
    """
    issues = []
    reqs = query_to_dicts(graph, ADCS_REQUIREMENTS)

    for req in reqs:
        name = req["name"]
        q = f"""
        SELECT (COUNT(?ev) AS ?cnt) WHERE {{
            ?ev rtm:addresses ?req .
            ?req sysml:declaredName "{name}" .
        }}
        """
        rows = query_to_dicts(graph, q)
        if rows and int(rows[0]["cnt"]) == 0:
            issues.append(f"{name}: no evidence artifacts found")

    return issues


def get_attestation_status(graph: Graph) -> list[dict[str, str]]:
    """Return attestation status for each ADCS requirement."""
    return query_to_dicts(graph, ATTESTATION_STATUS)


def get_unattested_requirements(graph: Graph) -> list[str]:
    """Return names of requirements that have not been attested."""
    rows = query_to_dicts(graph, UNATTESTED_REQUIREMENTS)
    return [r["reqName"] for r in rows]


def export_rtm(graph: Graph | Dataset, path: str | Path) -> None:
    """Serialize the RTM as Turtle (flat union if Dataset).

    Side effect: if `graph` is a Dataset, also writes a .trig file at
    the same base path so the named-graph structure is preserved.
    """
    from pipeline.dataset import export_trig, export_union_turtle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(graph, Dataset):
        export_union_turtle(graph, path)
        trig_path = path.with_suffix(".trig")
        export_trig(graph, trig_path)
    else:
        graph.serialize(str(path), format="turtle")


def print_rtm_summary(graph: Graph) -> str:
    """Format an RTM summary table as a string.

    Separates two orthogonal facts per requirement:
      - Whether an attestation EXISTS (trace integrity)
      - The attestation's earl outcome (engineering verdict)

    A requirement attested with `earl:failed` is structurally traced
    (the chain is intact) but not satisfied (engineering work needed).
    The previous version conflated these by reporting only "ATTESTED" /
    "UNATTESTED" — masking failed outcomes.
    """
    lines = []
    lines.append("=" * 72)
    lines.append("REQUIREMENTS TRACEABILITY MATRIX — STATUS SUMMARY")
    lines.append("=" * 72)

    reqs = query_to_dicts(graph, ADCS_REQUIREMENTS)
    outcomes = {
        r["reqName"]: r["outcomeShort"]
        for r in query_to_dicts(graph, REQUIREMENT_OUTCOMES)
    }

    needs_work: list[str] = []  # attested but earl:failed / earl:cantTell
    not_traced: list[str] = []  # no attestation at all

    for req in reqs:
        name = req["name"]
        text = req["text"].strip().replace("\n", " ")[:60]
        outcome = outcomes.get(name) or ""

        if outcome == "passed":
            status = "ATTESTED (earl:passed)"
        elif outcome == "failed":
            status = "ATTESTED (earl:failed) — engineering finding"
            needs_work.append(name)
        elif outcome == "cantTell":
            status = "ATTESTED (earl:cantTell) — insufficient evidence"
            needs_work.append(name)
        elif outcome == "inapplicable":
            status = "ATTESTED (earl:inapplicable)"
        elif outcome == "untested":
            status = "ATTESTED (earl:untested)"
        elif outcome:
            status = f"ATTESTED (earl:{outcome})"
        else:
            status = "NO ATTESTATION — trace incomplete"
            not_traced.append(name)

        lines.append(f"\n  {name}: {text}...")
        lines.append(f"    Status: {status}")

    lines.append("\n" + "=" * 72)
    if not_traced:
        lines.append(f"  Untraced (no attestation): {', '.join(not_traced)}")
    if needs_work:
        lines.append(f"  Open engineering findings (covered+failed/cantTell): "
                     f"{', '.join(needs_work)}")
    if not not_traced and not needs_work:
        lines.append("  All requirements attested with earl:passed.")
    lines.append("=" * 72)

    return "\n".join(lines)
