"""'How do you know X?' — walk the RTM graph and produce explanation chains.

Given a requirement name, walks the traceability graph and produces a
human-readable explanation chain showing:
- The requirement and its derivation
- Allocated design elements
- All evidence artifacts with summaries
- Attestation details (if attested)
- Live re-verification of proof evidence
"""

from __future__ import annotations

from rdflib import Graph

from analysis.proof_scripts import ProofStatus
from interrogate.reproduce import reproduce_proof
from ontology.prefixes import PROV, RTM, SYSML
from traceability.queries import query_to_dicts


def explain_requirement(graph: Graph, req_name: str) -> str:
    """Produce a full explanation chain for a requirement.

    Returns a formatted string showing the complete traceability chain.
    """
    lines = []

    # Requirement text
    q = f"""
    SELECT ?text WHERE {{
        ?req sysml:declaredName "{req_name}" ;
             sysml:text ?text .
    }}
    """
    rows = query_to_dicts(graph, q)
    if not rows:
        return f"Requirement '{req_name}' not found in graph."
    text = rows[0]["text"].strip().replace("\n", " ")
    lines.append(f'{req_name}: "{text}"')

    # Derived from
    q = f"""
    SELECT ?parentName ?parentText WHERE {{
        ?req sysml:declaredName "{req_name}" ;
             rtm:derivedFrom ?parent .
        ?parent sysml:declaredName ?parentName .
        OPTIONAL {{ ?parent sysml:text ?parentText }}
    }}
    """
    derived = query_to_dicts(graph, q)
    if derived:
        lines.append(f"├── Derived from: {derived[0]['parentName']} (satellite-level)")

    # Allocated design elements
    q = f"""
    SELECT ?elementName WHERE {{
        ?req sysml:declaredName "{req_name}" ;
             sysml:ownedRelationship ?rel .
        ?rel a sysml:SatisfyRequirementUsage ;
             sysml:satisfyingElement ?el .
        ?el sysml:declaredName ?elementName .
    }}
    ORDER BY ?elementName
    """
    alloc = query_to_dicts(graph, q)
    if alloc:
        elements = ", ".join(r["elementName"] for r in alloc)
        lines.append(f"├── Allocated to: {elements}")

    # Evidence artifacts
    q = f"""
    SELECT ?ev ?type ?hash ?summary ?proofHash ?modelHash ?sourceFile WHERE {{
        ?ev a ?type ;
            rtm:contentHash ?hash ;
            rtm:resultSummary ?summary ;
            rtm:addresses ?req .
        ?req sysml:declaredName "{req_name}" .
        FILTER(?type IN (rtm:ProofArtifact, rtm:SimulationResult))
        OPTIONAL {{ ?ev rtm:proofHash ?proofHash }}
        OPTIONAL {{ ?ev rtm:modelHash ?modelHash }}
        OPTIONAL {{ ?ev rtm:sourceFile ?sourceFile }}
    }}
    ORDER BY ?type
    """
    evidence = query_to_dicts(graph, q)
    if evidence:
        lines.append(f"├── Evidence ({len(evidence)} artifacts):")
        for ev in evidence:
            ev_type = ev["type"].split("#")[-1]
            prefix = "│   ├──" if ev != evidence[-1] else "│   └──"
            lines.append(f"{prefix} [{ev_type}]")
            lines.append(f"│   │   content_hash: {ev['hash'][:16]}...")
            if ev.get("proofHash"):
                lines.append(f"│   │   proof_hash: {ev['proofHash'][:16]}...")
            if ev.get("modelHash"):
                lines.append(f"│   │   model_hash: {ev['modelHash'][:16]}...")

            # Re-verify proof evidence live
            if ev_type == "ProofArtifact" and ev.get("modelHash"):
                rv = reproduce_proof(graph, ev["ev"])
                if rv is not None:
                    status_str = "VERIFIED" if rv["status"] == ProofStatus.VERIFIED else "FAILED"
                    lines.append(f"│   │   Re-verification: {status_str} (re-executed just now)")
                    for lr in rv.get("lemma_results", []):
                        mark = "✓" if lr["passed"] else "✗"
                        lines.append(f"│   │     {lr['name']}: {mark}")

            lines.append(f"│   │   {ev['summary']}")
            if ev.get("sourceFile"):
                lines.append(f"│   │   source: {ev['sourceFile']}")
    else:
        lines.append("├── Evidence: NONE")

    # Attestation
    q = f"""
    SELECT ?engineer ?adequacy ?sufficiency ?timestamp ?gitCommit WHERE {{
        ?att a rtm:Attestation ;
             rtm:attests ?req ;
             rtm:modelAdequacy ?adequacy ;
             rtm:evidenceSufficiency ?sufficiency ;
             prov:wasAssociatedWith ?agent ;
             prov:generatedAtTime ?timestamp .
        ?agent rdfs:label ?engineer .
        ?req sysml:declaredName "{req_name}" .
        OPTIONAL {{ ?att rtm:gitCommit ?gitCommit }}
    }}
    """
    attestations = query_to_dicts(graph, q)
    if attestations:
        att = attestations[0]
        lines.append(f"└── Attestation:")
        lines.append(f"    Attested by: {att['engineer']}")
        lines.append(f"    Timestamp: {att['timestamp']}")
        if att.get("gitCommit"):
            lines.append(f"    Git commit: {att['gitCommit'][:12]}")
        lines.append(f"    Model adequacy: \"{att['adequacy']}\"")
        lines.append(f"    Evidence sufficiency: \"{att['sufficiency']}\"")
    else:
        lines.append("└── Attestation: NOT ATTESTED")

    return "\n".join(lines)


def explain_all(graph: Graph) -> str:
    """Produce explanation chains for all ADCS requirements."""
    from traceability.queries import ADCS_REQUIREMENTS
    reqs = query_to_dicts(graph, ADCS_REQUIREMENTS)

    sections = []
    for req in reqs:
        sections.append(explain_requirement(graph, req["name"]))

    separator = "\n" + "=" * 60 + "\n"
    return separator.join(sections)
