"""Fetch upstream ontologies into ontology/imports/.

Re-runnable / idempotent. Each upstream is fetched at its published IRI,
parsed via rdflib (handles RDF/XML, Turtle, N3), and re-serialized as
Turtle into ontology/imports/<name>.ttl.

The vendored copies are committed to git so demo reproducibility doesn't
depend on upstream IRIs continuing to resolve.

The full OMG SysMLv2 OWL rendering is deliberately not vendored here —
it is distributed via Maven and requires a separate Java/Maven
toolchain to extract. The equivalence axioms in rtm-edit.ttl
(sourced from sysml_term_map.csv) bind to omg-sysml: terms
regardless; vendoring the full upstream is future work.

Usage:
    uv run python -m scripts.fetch_imports
"""

from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from rdflib import Graph

IMPORTS_DIR = Path(__file__).resolve().parent.parent / "ontology" / "imports"


@dataclass(frozen=True)
class Source:
    name: str
    iri: str
    fetch_url: str
    format: str          # rdflib parser format: "turtle", "xml", ...
    output: str          # filename under ontology/imports/


SOURCES: list[Source] = [
    Source(
        name="PROV-O",
        iri="http://www.w3.org/ns/prov-o",
        fetch_url="https://www.w3.org/ns/prov-o.ttl",
        format="turtle",
        output="prov-o.ttl",
    ),
    Source(
        name="EARL",
        iri="http://www.w3.org/ns/earl#",
        fetch_url="https://www.w3.org/ns/earl.rdf",
        format="xml",
        output="earl.ttl",
    ),
    Source(
        name="OntoGSN",
        iri="https://w3id.org/OntoGSN/ontology",
        fetch_url="https://raw.githubusercontent.com/fortiss/OntoGSN/main/serializations/ontogsn.ttl",
        format="turtle",
        output="ontogsn.ttl",
    ),
    Source(
        name="P-PLAN",
        iri="http://purl.org/net/p-plan",
        fetch_url="https://vocab.linkeddata.es/p-plan/",
        format="xml",
        output="p-plan.ttl",
    ),
    Source(
        name="OSLC RM",
        iri="http://open-services.net/ns/rm#",
        fetch_url="https://raw.githubusercontent.com/oasis-tcs/oslc-domains/master/rm/requirements-management-vocab.ttl",
        format="turtle",
        output="oslc-rm.ttl",
    ),
    Source(
        name="OSLC QM",
        iri="http://open-services.net/ns/qm#",
        fetch_url="https://raw.githubusercontent.com/oasis-tcs/oslc-domains/master/qm/quality-management-vocab.ttl",
        format="turtle",
        output="oslc-qm.ttl",
    ),
]


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "text/turtle, application/rdf+xml;q=0.9, */*;q=0.5",
            "User-Agent": "adcs-lifecycle-demo/fetch_imports.py",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read()


def fetch_one(source: Source) -> tuple[Source, int]:
    """Fetch and re-serialize one upstream ontology. Returns (source, triple count)."""
    raw = _fetch(source.fetch_url)
    g = Graph()
    g.parse(data=raw, format=source.format, publicID=source.iri)
    out_path = IMPORTS_DIR / source.output
    g.serialize(destination=out_path, format="turtle")
    return source, len(g)


def fetch_all() -> int:
    IMPORTS_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for src in SOURCES:
        try:
            _, triples = fetch_one(src)
            print(f"  OK  {src.name:12} {triples:>5} triples -> {src.output}")
        except Exception as exc:
            print(f"  FAIL {src.name:12} {exc}", file=sys.stderr)
            failures += 1
    return failures


if __name__ == "__main__":
    print(f"Fetching upstream ontologies into {IMPORTS_DIR.relative_to(Path.cwd())}/ ...")
    rc = fetch_all()
    sys.exit(1 if rc else 0)
