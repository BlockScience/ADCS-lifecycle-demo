"""Named-graph quadstore helpers.

The runtime moves from a single rdflib.Graph to an rdflib.Dataset with
eight named graphs sized to match how Flexo MMS partitions projects/
branches/refs. See plan in /Users/z/.claude/plans/i-want-to-look-hidden-balloon.md
section "Named graphs / quadstore layout" for the full design.

Named graph IRIs are exported as constants by ontology.prefixes
(G_ONTOLOGY, G_STRUCTURAL, G_EVIDENCE, ...). This module gives the
runtime a small surface over the Dataset:

- create_dataset()             build a Dataset with prefixes bound and
                               default_union=True so existing SPARQL
                               queries continue to work without GRAPH
                               clauses.
- graph_for(ds, layer)         return the named-graph view for that
                               layer; writes go to the corresponding
                               quad slot.
- load_into(ds, layer, path)   parse a TTL file into that layer.
- triples_by_graph(ds)         {graph_iri: count} — sanity / display.
- export_trig(ds, path)        serialize the full quadstore as TriG.
- export_union_turtle(ds, p)   serialize the union view as flat Turtle
                               (back-compat with the existing rtm.ttl
                               export that downstream tools consume).
"""

from __future__ import annotations

from pathlib import Path

from rdflib import Dataset, Graph, URIRef

from ontology.prefixes import NAMED_GRAPHS, bind_prefixes


def create_dataset() -> Dataset:
    """Build a Dataset with prefixes bound and default_union=True.

    default_union=True is load-bearing: it makes `dataset.query(sparql)`
    match triples across every named graph when no GRAPH clause is used.
    Without this, the pipeline's existing SPARQL queries would return
    empty result sets after the migration.
    """
    ds = Dataset(default_union=True)
    bind_prefixes(ds)
    return ds


def graph_for(ds: Dataset, layer: str) -> Graph:
    """Return the named-graph view for `layer` (e.g. "evidence").

    The returned Graph is a live view backed by the Dataset's quad
    store; triples added to it are stored as quads in that named graph.
    """
    if layer not in NAMED_GRAPHS:
        raise KeyError(
            f"Unknown named-graph layer {layer!r}. "
            f"Valid layers: {sorted(NAMED_GRAPHS)}"
        )
    return ds.graph(URIRef(NAMED_GRAPHS[layer]))


def load_into(ds: Dataset, layer: str, ttl_path: str | Path) -> int:
    """Parse a Turtle file into the named graph for `layer`. Returns triples added."""
    g = graph_for(ds, layer)
    before = len(g)
    g.parse(str(ttl_path), format="turtle")
    return len(g) - before


def query_named_graph(ds: Dataset, layer: str, sparql: str, **bindings):
    """Run a SPARQL query scoped to a single named graph.

    Use this when the query is intentionally layer-scoped (e.g. "count
    attestations only"). The default query path — `ds.query(sparql)` —
    walks the union view because the runtime constructs Datasets with
    `default_union=True`; use that when the query is meant to join
    across named graphs.

    `bindings` are forwarded as initBindings.

    Raises KeyError if `layer` is not a known named-graph name.
    """
    if layer not in NAMED_GRAPHS:
        raise KeyError(
            f"Unknown named-graph layer {layer!r}. "
            f"Valid layers: {sorted(NAMED_GRAPHS)}"
        )
    g = ds.graph(URIRef(NAMED_GRAPHS[layer]))
    return g.query(sparql, initBindings=bindings)


def triples_by_graph(ds: Dataset) -> dict[str, int]:
    """Return {graph_iri: triple_count} for every populated named graph."""
    counts: dict[str, int] = {}
    for ctx in ds.contexts():
        ctx_iri = str(ctx.identifier)
        # rdflib's Dataset uses a sentinel default-graph identifier; skip empty contexts.
        n = len(ctx)
        if n:
            counts[ctx_iri] = n
    return counts


def export_trig(ds: Dataset, path: str | Path) -> None:
    """Serialize the entire Dataset as TriG (multi-graph Turtle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.serialize(destination=str(path), format="trig")


def export_union_turtle(ds: Dataset, path: str | Path) -> None:
    """Serialize the union of all named graphs as flat Turtle.

    Preserves backward compatibility with the existing rtm.ttl export
    that downstream tools (interrogate, visualize, external consumers)
    expect.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    union = Graph()
    bind_prefixes(union)
    for s, p, o in ds.triples((None, None, None)):
        union.add((s, p, o))
    union.serialize(destination=str(path), format="turtle")
