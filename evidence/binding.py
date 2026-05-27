"""Bind computational evidence to the RDF traceability graph.

Creates rtm:Evidence nodes (ProofArtifact, SimulationResult) linked to:
- The computational activity that produced them (prov:wasGeneratedBy)
- The requirement they address (rtm:addresses) — structural intent, not judgment

Evidence *addresses* a requirement but does not *satisfy* it.
Only human attestation (rtm:attests) connects evidence to satisfaction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from ontology.prefixes import ADCS, PROV, RTM, bind_prefixes

if TYPE_CHECKING:
    from compute.base import ExecutionMetadata


def _bind_execution_metadata(
    graph: Graph,
    activity_uri: URIRef,
    metadata: "ExecutionMetadata | None",
) -> None:
    """Attach execution-context PROV triples to an analysis activity.

    Records WHERE the analysis ran (location kind + hostname), and HOW
    (image digest + container ID for Docker-emulated remote compute, or
    the bare host's Python version for local compute). This is the
    audit trail's answer to "which physical/virtual machine produced
    this proof, with what toolchain version?"

    Emitted shape:
      <activity>
          prov:atLocation <urn:adcs:location:...> ;
          prov:wasAssociatedWith <urn:adcs:executor:...> .

      <urn:adcs:executor:...> a prov:SoftwareAgent ;
          rtm:hostname "..." ;
          rtm:imageDigest "sha256:..." ;   # Docker only
          rtm:containerId "..." ;          # Docker only
          rtm:pythonVersion "..." .

      <urn:adcs:location:...> a prov:Location ;
          rdfs:label "..." .
    """
    if metadata is None:
        return

    # Executor + location IRI construction lives on ExecutionMetadata so
    # WP3 / WP4 can reuse the same shapes when introducing new evidence
    # types (rtm:DockerImage, three-remote URIs).
    executor = metadata.executor_uri()
    location = metadata.location_uri()

    graph.add((activity_uri, PROV.atLocation, location))
    graph.add((activity_uri, PROV.wasAssociatedWith, executor))

    graph.add((executor, RDF.type, PROV.SoftwareAgent))
    if metadata.hostname:
        graph.add((executor, RTM.hostname, Literal(metadata.hostname)))
    if metadata.image_digest:
        graph.add((executor, RTM.imageDigest, Literal(metadata.image_digest)))
    if metadata.image_label:
        graph.add((executor, RTM.imageLabel, Literal(metadata.image_label)))
    if metadata.container_id:
        graph.add((executor, RTM.containerId, Literal(metadata.container_id)))
    if metadata.python_version:
        graph.add((executor, RTM.pythonVersion, Literal(metadata.python_version)))
    graph.add((executor, RDFS.label,
               Literal(f"{metadata.location_kind} executor on {metadata.hostname or '?'}")))

    graph.add((location, RDF.type, PROV.Location))
    graph.add((location, RDFS.label,
               Literal(f"{metadata.location_kind}:{metadata.hostname or '?'}")))

    if metadata.started_at:
        graph.add((activity_uri, PROV.startedAtTime,
                   Literal(metadata.started_at, datatype=XSD.dateTime)))
    if metadata.ended_at:
        graph.add((activity_uri, PROV.endedAtTime,
                   Literal(metadata.ended_at, datatype=XSD.dateTime)))


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
    execution_metadata: "ExecutionMetadata | None" = None,
) -> URIRef:
    """Create an rtm:ProofArtifact node in the graph.

    If execution_metadata is provided, the analysis activity is
    additionally annotated with where (prov:atLocation) and how
    (image / hostname / container ID) it ran — the RTM provenance for
    "this proof was produced on remote-compute server X using image Y".

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
    _bind_execution_metadata(graph, act_uri, execution_metadata)

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
    execution_metadata: "ExecutionMetadata | None" = None,
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
    _bind_execution_metadata(graph, act_uri, execution_metadata)

    return ev_uri


def bind_computation_engines(graph: Graph) -> None:
    """Add agent nodes for the computation engines."""
    sympy_uri = ADCS["SymPyEngine"]
    scipy_uri = ADCS["ScipyEngine"]

    graph.add((sympy_uri, RDF.type, RTM.ComputationEngine))
    graph.add((sympy_uri, PROV.label, Literal("SymPy symbolic engine")))

    graph.add((scipy_uri, RDF.type, RTM.ComputationEngine))
    graph.add((scipy_uri, PROV.label, Literal("SciPy numerical engine")))
