"""RDF namespace constants for the ADCS lifecycle demo.

The RTM ontology is an integration / application ontology — a thin layer
assembled from established upstream vocabularies. Custom rtm: terms are
limited to convenience subclasses, content-addressing properties, and
SHACL targets. Every epistemic concept comes from an imported standard.

Upstream vocabularies:
- PROV-O      : provenance spine
- SysMLv2     : structural model + requirements (local namespace, aliased
                to the OMG SysMLv2 OWL rendering via equivalence axioms)
- EARL        : assertion pattern + outcome lattice + mode
- OntoGSN     : Goal / Strategy / Solution / Assumption / Justification / Context
- P-PLAN      : declarative pipeline plan + step ordering
- OSLC RM/QM  : tool-interop aliases for requirement satisfaction / validation
- SHACL       : closure rules (well-formedness invariants)
- Dublin Core : metadata
"""

from rdflib import Namespace

# Layer 1 — W3C / IETF standards
PROV = Namespace("http://www.w3.org/ns/prov#")
DCTERMS = Namespace("http://purl.org/dc/terms/")
EARL = Namespace("http://www.w3.org/ns/earl#")
SH = Namespace("http://www.w3.org/ns/shacl#")

# Layer 1 — Community ontologies
GSN = Namespace("https://w3id.org/OntoGSN/ontology#")
P_PLAN = Namespace("http://purl.org/net/p-plan#")
OSLC_RM = Namespace("http://open-services.net/ns/rm#")
OSLC_QM = Namespace("http://open-services.net/ns/qm#")

# Layer 2 — SysMLv2
# Local namespace used throughout instance data. Equivalence axioms in
# rtm-edit.ttl bind these terms to the OMG SysMLv2 OWL rendering so
# JPL/OpenMBEE-aware tooling sees standard SysMLv2.
SYSML = Namespace("https://www.omg.org/spec/SysML/2.0/")
OMG_SYSML = Namespace("http://www.omg.org/spec/SysML/20240501/")  # OMG SysMLv2 OWL alias target

# Layer 3 — Local integration vocabulary + instance namespaces
RTM = Namespace("http://example.org/ontology/rtm#")
ADCS = Namespace("http://example.org/adcs-demo/")
SAT = Namespace("http://example.org/adcs-demo/satellite/")

# Named-graph IRIs (Flexo-compatible quadstore layout)
G_ONTOLOGY = "http://example.org/adcs-demo/graph/ontology"
G_PLAN = "http://example.org/adcs-demo/graph/plan"
G_STRUCTURAL = "http://example.org/adcs-demo/graph/structural"
G_CONTEXT = "http://example.org/adcs-demo/graph/context"
G_EVIDENCE = "http://example.org/adcs-demo/graph/evidence"
G_ATTESTATIONS = "http://example.org/adcs-demo/graph/attestations"
G_PLAN_EXECUTION = "http://example.org/adcs-demo/graph/plan-execution"
G_AUDIT = "http://example.org/adcs-demo/graph/audit"

NAMED_GRAPHS = {
    "ontology": G_ONTOLOGY,
    "plan": G_PLAN,
    "structural": G_STRUCTURAL,
    "context": G_CONTEXT,
    "evidence": G_EVIDENCE,
    "attestations": G_ATTESTATIONS,
    "plan_execution": G_PLAN_EXECUTION,
    "audit": G_AUDIT,
}

# All prefixes for graph binding
PREFIXES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "prov": str(PROV),
    "dcterms": str(DCTERMS),
    "earl": str(EARL),
    "sh": str(SH),
    "gsn": str(GSN),
    "p-plan": str(P_PLAN),
    "oslc_rm": str(OSLC_RM),
    "oslc_qm": str(OSLC_QM),
    "sysml": str(SYSML),
    "omg-sysml": str(OMG_SYSML),
    "rtm": str(RTM),
    "adcs": str(ADCS),
    "sat": str(SAT),
}


def bind_prefixes(graph):
    """Bind all project prefixes to an rdflib Graph or Dataset."""
    for prefix, uri in PREFIXES.items():
        graph.bind(prefix, uri)
    return graph
