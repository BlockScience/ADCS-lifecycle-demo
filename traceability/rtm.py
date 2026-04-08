"""Requirements Traceability Matrix assembly and validation.

Merges the structural graph with evidence and attestation graphs,
validates completeness, and exports the full RTM.
"""

from __future__ import annotations

from pathlib import Path

from rdflib import Graph

from ontology.prefixes import ADCS, PROV, RTM, SYSML, bind_prefixes
from traceability.queries import (
    ADCS_REQUIREMENTS,
    ATTESTATION_STATUS,
    UNATTESTED_REQUIREMENTS,
    query_to_dicts,
)

ONTOLOGY_DIR = Path(__file__).resolve().parent.parent / "ontology"
STRUCTURAL_DIR = Path(__file__).resolve().parent.parent / "structural"


def load_base_graph() -> Graph:
    """Load ontology + structural model into a graph."""
    g = Graph()
    bind_prefixes(g)
    for d in [ONTOLOGY_DIR, STRUCTURAL_DIR]:
        for ttl in sorted(d.glob("*.ttl")):
            g.parse(ttl, format="turtle")
    return g


def assemble_rtm(
    base_graph: Graph,
    evidence_graph: Graph,
) -> Graph:
    """Merge base (structural + ontology) with evidence into a single RTM graph."""
    rtm = Graph()
    bind_prefixes(rtm)

    # Copy all triples from both graphs
    for triple in base_graph:
        rtm.add(triple)
    for triple in evidence_graph:
        rtm.add(triple)

    return rtm


def validate_structural_completeness(graph: Graph) -> list[str]:
    """Check that every ADCS requirement has at least one satisfy link.

    Returns a list of issues (empty if all good).
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


def validate_evidence_completeness(graph: Graph) -> list[str]:
    """Check that every ADCS requirement has associated evidence.

    Returns a list of issues (empty if all good).
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


def export_rtm(graph: Graph, path: str | Path) -> None:
    """Serialize the RTM graph as Turtle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(str(path), format="turtle")


def print_rtm_summary(graph: Graph) -> str:
    """Format an RTM summary table as a string."""
    lines = []
    lines.append("=" * 72)
    lines.append("REQUIREMENTS TRACEABILITY MATRIX — STATUS SUMMARY")
    lines.append("=" * 72)

    reqs = query_to_dicts(graph, ADCS_REQUIREMENTS)
    status = {r["reqName"]: r["attestCount"] for r in get_attestation_status(graph)}

    for req in reqs:
        name = req["name"]
        text = req["text"].strip().replace("\n", " ")[:60]
        count = status.get(name, "0")
        attested = "ATTESTED" if int(count) > 0 else "UNATTESTED"
        lines.append(f"\n  {name}: {text}...")
        lines.append(f"    Status: {attested}")

    lines.append("\n" + "=" * 72)
    unattested = get_unattested_requirements(graph)
    if unattested:
        lines.append(f"  Unattested: {', '.join(unattested)}")
    else:
        lines.append("  All requirements attested.")
    lines.append("=" * 72)

    return "\n".join(lines)
