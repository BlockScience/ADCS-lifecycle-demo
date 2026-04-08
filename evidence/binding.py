"""Bind computational evidence to the RDF traceability graph.

Creates rtm:Evidence nodes (ProofArtifact, SimulationResult) linked to:
- The computational activity that produced them (prov:wasGeneratedBy)
- The requirement they address (rtm:addresses) — structural intent, not judgment

Evidence *addresses* a requirement but does not *satisfy* it.
Only human attestation (rtm:attests) connects evidence to satisfaction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from ontology.prefixes import ADCS, PROV, RTM, bind_prefixes


def bind_proof_evidence(
    graph: Graph,
    evidence_id: str,
    activity_id: str,
    requirement_id: str,
    model_hash: str,
    proof_hash: str,
    content_hash: str,
    result_summary: str,
    source_file: str = "",
    git_commit: str = "",
) -> URIRef:
    """Create an rtm:ProofArtifact node in the graph.

    Returns the URI of the new evidence node.
    """
    ev_uri = ADCS[evidence_id]
    act_uri = ADCS[activity_id]

    graph.add((ev_uri, RDF.type, RTM.ProofArtifact))
    graph.add((ev_uri, RTM.contentHash, Literal(content_hash)))
    graph.add((ev_uri, RTM.modelHash, Literal(model_hash)))
    graph.add((ev_uri, RTM.proofHash, Literal(proof_hash)))
    graph.add((ev_uri, RTM.resultSummary, Literal(result_summary)))
    graph.add((ev_uri, RTM.evidenceMethod, RTM.FormalProof))
    graph.add((ev_uri, RTM.addresses, ADCS[requirement_id]))
    graph.add((ev_uri, PROV.wasGeneratedBy, act_uri))
    graph.add((ev_uri, PROV.generatedAtTime, Literal(
        datetime.now(timezone.utc).isoformat(), datatype=XSD.dateTime,
    )))

    if source_file:
        graph.add((ev_uri, RTM.sourceFile, Literal(source_file)))
    if git_commit:
        graph.add((ev_uri, RTM.gitCommit, Literal(git_commit)))

    # Activity node
    graph.add((act_uri, RDF.type, RTM.SymbolicAnalysis))
    graph.add((act_uri, PROV.used, ADCS[requirement_id]))
    graph.add((act_uri, PROV.wasAssociatedWith, ADCS["SymPyEngine"]))

    return ev_uri


def bind_simulation_evidence(
    graph: Graph,
    evidence_id: str,
    activity_id: str,
    requirement_id: str,
    model_hash: str,
    sim_hash: str,
    result_summary: str,
    sim_config: dict[str, Any] | None = None,
    source_file: str = "",
    git_commit: str = "",
) -> URIRef:
    """Create an rtm:SimulationResult node in the graph.

    Returns the URI of the new evidence node.
    """
    ev_uri = ADCS[evidence_id]
    act_uri = ADCS[activity_id]

    graph.add((ev_uri, RDF.type, RTM.SimulationResult))
    graph.add((ev_uri, RTM.contentHash, Literal(sim_hash)))
    graph.add((ev_uri, RTM.modelHash, Literal(model_hash)))
    graph.add((ev_uri, RTM.resultSummary, Literal(result_summary)))
    graph.add((ev_uri, RTM.evidenceMethod, RTM.Simulation))
    graph.add((ev_uri, RTM.addresses, ADCS[requirement_id]))
    graph.add((ev_uri, PROV.wasGeneratedBy, act_uri))
    graph.add((ev_uri, PROV.generatedAtTime, Literal(
        datetime.now(timezone.utc).isoformat(), datatype=XSD.dateTime,
    )))

    if source_file:
        graph.add((ev_uri, RTM.sourceFile, Literal(source_file)))
    if git_commit:
        graph.add((ev_uri, RTM.gitCommit, Literal(git_commit)))

    # Activity node
    graph.add((act_uri, RDF.type, RTM.NumericalSimulation))
    graph.add((act_uri, PROV.used, ADCS[requirement_id]))
    graph.add((act_uri, PROV.wasAssociatedWith, ADCS["ScipyEngine"]))

    return ev_uri


def bind_computation_engines(graph: Graph) -> None:
    """Add agent nodes for the computation engines."""
    sympy_uri = ADCS["SymPyEngine"]
    scipy_uri = ADCS["ScipyEngine"]

    graph.add((sympy_uri, RDF.type, RTM.ComputationEngine))
    graph.add((sympy_uri, PROV.label, Literal("SymPy symbolic engine")))

    graph.add((scipy_uri, RDF.type, RTM.ComputationEngine))
    graph.add((scipy_uri, PROV.label, Literal("SciPy numerical engine")))
