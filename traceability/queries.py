"""SPARQL query templates for RTM interrogation.

All queries use the project namespace prefixes. Results are returned as
lists of dicts for easy consumption.
"""

from __future__ import annotations

from rdflib import Graph

from ontology.prefixes import ADCS, PROV, RTM, SYSML

_INIT_NS = {"sysml": SYSML, "rtm": RTM, "prov": PROV, "adcs": ADCS}

# ---------------------------------------------------------------------------
# Requirement queries
# ---------------------------------------------------------------------------

ADCS_REQUIREMENTS = """
SELECT ?req ?name ?text WHERE {
    ?req a sysml:RequirementDefinition ;
         sysml:declaredName ?name ;
         sysml:text ?text .
    FILTER(STRSTARTS(?name, "REQ-"))
}
ORDER BY ?name
"""

REQUIREMENT_ALLOCATION = """
SELECT ?reqName ?elementName WHERE {
    ?req a sysml:RequirementDefinition ;
         sysml:declaredName ?reqName ;
         sysml:ownedRelationship ?rel .
    ?rel a sysml:SatisfyRequirementUsage ;
         sysml:satisfyingElement ?element .
    ?element sysml:declaredName ?elementName .
    FILTER(STRSTARTS(?reqName, "REQ-"))
}
ORDER BY ?reqName ?elementName
"""

REQUIREMENT_DERIVATION = """
SELECT ?adcsName ?satName WHERE {
    ?req sysml:declaredName ?adcsName ;
         rtm:derivedFrom ?parent .
    ?parent sysml:declaredName ?satName .
}
ORDER BY ?adcsName
"""

# ---------------------------------------------------------------------------
# Evidence queries
# ---------------------------------------------------------------------------

ALL_EVIDENCE = """
SELECT ?ev ?type ?method ?hash ?summary WHERE {
    ?ev a ?type ;
        rtm:contentHash ?hash ;
        rtm:resultSummary ?summary .
    OPTIONAL { ?ev rtm:evidenceMethod ?method }
    FILTER(?type IN (rtm:ProofArtifact, rtm:SimulationResult))
}
ORDER BY ?ev
"""

EVIDENCE_FOR_REQUIREMENT = """
SELECT ?ev ?type ?hash ?summary WHERE {
    ?ev a ?type ;
        rtm:contentHash ?hash ;
        rtm:resultSummary ?summary ;
        rtm:addresses ?req .
    ?req sysml:declaredName ?reqName .
    FILTER(?type IN (rtm:ProofArtifact, rtm:SimulationResult))
    FILTER(?reqName = "%s")
}
ORDER BY ?ev
"""

# ---------------------------------------------------------------------------
# Attestation queries
# ---------------------------------------------------------------------------

ALL_ATTESTATIONS = """
SELECT ?att ?reqName ?engineer ?adequacy ?sufficiency ?timestamp WHERE {
    ?att a rtm:Attestation ;
         rtm:attests ?req ;
         rtm:modelAdequacy ?adequacy ;
         rtm:evidenceSufficiency ?sufficiency ;
         prov:wasAssociatedWith ?agent ;
         prov:generatedAtTime ?timestamp .
    ?req sysml:declaredName ?reqName .
    ?agent rdfs:label ?engineer .
}
ORDER BY ?reqName
"""

ATTESTATION_STATUS = """
SELECT ?reqName (COUNT(?att) AS ?attestCount) WHERE {
    ?req a sysml:RequirementDefinition ;
         sysml:declaredName ?reqName .
    FILTER(STRSTARTS(?reqName, "REQ-"))
    OPTIONAL {
        ?att a rtm:Attestation ;
             rtm:attests ?req .
    }
}
GROUP BY ?reqName
ORDER BY ?reqName
"""

UNATTESTED_REQUIREMENTS = """
SELECT ?reqName WHERE {
    ?req a sysml:RequirementDefinition ;
         sysml:declaredName ?reqName .
    FILTER(STRSTARTS(?reqName, "REQ-"))
    FILTER NOT EXISTS {
        ?att a rtm:Attestation ;
             rtm:attests ?req .
    }
}
ORDER BY ?reqName
"""

# ---------------------------------------------------------------------------
# Forward / backward trace
# ---------------------------------------------------------------------------

FORWARD_TRACE = """
SELECT ?reqName ?elementName ?evType ?evHash ?evSummary ?attEngineer ?attTime WHERE {
    ?req a sysml:RequirementDefinition ;
         sysml:declaredName ?reqName ;
         sysml:ownedRelationship ?rel .
    ?rel a sysml:SatisfyRequirementUsage ;
         sysml:satisfyingElement ?element .
    ?element sysml:declaredName ?elementName .
    FILTER(?reqName = "%s")

    OPTIONAL {
        ?ev a ?evType ;
            rtm:contentHash ?evHash ;
            rtm:resultSummary ?evSummary ;
            rtm:addresses ?req .
        FILTER(?evType IN (rtm:ProofArtifact, rtm:SimulationResult))
    }
    OPTIONAL {
        ?att a rtm:Attestation ;
             rtm:attests ?req ;
             prov:wasAssociatedWith ?agent ;
             prov:generatedAtTime ?attTime .
        ?agent rdfs:label ?attEngineer .
    }
}
ORDER BY ?elementName ?evType
"""

BACKWARD_TRACE = """
SELECT ?reqName ?evHash ?evSummary ?activityType ?attEngineer ?attTime WHERE {
    ?att a rtm:Attestation ;
         rtm:attests ?req ;
         rtm:hasEvidence ?ev ;
         prov:wasAssociatedWith ?agent ;
         prov:generatedAtTime ?attTime .
    ?agent rdfs:label ?attEngineer .
    ?req sysml:declaredName ?reqName .
    ?ev rtm:contentHash ?evHash ;
        rtm:resultSummary ?evSummary ;
        prov:wasGeneratedBy ?activity .
    ?activity a ?activityType .
    FILTER(?activityType IN (rtm:SymbolicAnalysis, rtm:NumericalSimulation))
}
ORDER BY ?reqName ?evHash
"""

# ---------------------------------------------------------------------------
# Hash chain validation
# ---------------------------------------------------------------------------

EVIDENCE_HASH_CHAIN = """
SELECT ?ev ?modelHash ?proofHash ?contentHash WHERE {
    ?ev a rtm:ProofArtifact ;
        rtm:modelHash ?modelHash ;
        rtm:proofHash ?proofHash ;
        rtm:contentHash ?contentHash .
}
ORDER BY ?ev
"""


# ---------------------------------------------------------------------------
# Query runner helpers
# ---------------------------------------------------------------------------

def query_to_dicts(graph: Graph, sparql: str) -> list[dict[str, str]]:
    """Execute a SPARQL query and return results as list of dicts."""
    results = graph.query(sparql, initNs=_INIT_NS)
    rows = []
    for row in results:
        d = {}
        for var in results.vars:
            val = getattr(row, str(var), None)
            d[str(var)] = str(val) if val is not None else None
        rows.append(d)
    return rows
