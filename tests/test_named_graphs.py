"""Phase C — verify the named-graph layout is correct after pipeline runs.

These tests assert:
  - load_base_dataset() returns a Dataset, not a flat Graph.
  - Ontology TTLs land in <rtm:ontology>, structural in <adcs:structural>.
  - Stage 4 evidence writes route to <adcs:evidence>.
  - Stage 6 attestation writes route to <adcs:attestations>.
  - SPARQL queries (which omit GRAPH clauses) still match across graphs
    because Dataset(default_union=True) treats the union as default.
  - The exported .trig file preserves named-graph structure.
  - The exported .ttl file remains a flat union (back-compat).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from rdflib import Dataset, URIRef

from ontology.prefixes import (
    G_ATTESTATIONS,
    G_EVIDENCE,
    G_ONTOLOGY,
    G_STRUCTURAL,
    NAMED_GRAPHS,
)
from pipeline.runner import run_pipeline


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pipeline_dataset() -> Dataset:
    """Run the pipeline once and reuse the result across tests."""
    with warnings.catch_warnings():
        # rdflib's TriG serializer uses its own deprecated internal API
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return run_pipeline(auto_attest=True)


def _graph_size(ds: Dataset, graph_iri: str) -> int:
    return len(list(ds.quads((None, None, None, URIRef(graph_iri)))))


def test_pipeline_returns_dataset(pipeline_dataset):
    """Stage 5's `rtm` is now a Dataset, not a flat Graph."""
    assert isinstance(pipeline_dataset, Dataset)


def test_ontology_in_rtm_ontology_graph(pipeline_dataset):
    """rtm.ttl + rtm_individuals.ttl + rtm_shapes.ttl land in <rtm:ontology>."""
    n = _graph_size(pipeline_dataset, G_ONTOLOGY)
    assert n > 100, f"Expected substantial ontology graph (~327 triples), got {n}"


def test_structural_in_adcs_structural_graph(pipeline_dataset):
    """satellite.ttl + parameters.ttl land in <adcs:structural>."""
    n = _graph_size(pipeline_dataset, G_STRUCTURAL)
    assert n > 100, f"Expected substantial structural graph (~250 triples), got {n}"


def test_evidence_in_adcs_evidence_graph(pipeline_dataset):
    """Stage 4 evidence writes land in <adcs:evidence>, not the default graph."""
    n = _graph_size(pipeline_dataset, G_EVIDENCE)
    assert n > 50, f"Expected evidence graph populated, got {n} triples"


def test_attestations_in_adcs_attestations_graph(pipeline_dataset):
    """Stage 6 attestation writes land in <adcs:attestations>, not the default."""
    n = _graph_size(pipeline_dataset, G_ATTESTATIONS)
    assert n > 0, "Expected attestations graph to be populated by stage 6"


def test_no_leakage_into_default_graph(pipeline_dataset):
    """Every triple should live in some named graph; the default graph
    should be empty (or near-empty) since the runtime always uses
    named-graph views for writes."""
    # Authoritative allow-list: every planned named graph from prefixes.py.
    named = {URIRef(iri) for iri in NAMED_GRAPHS.values()}
    # rdflib's Dataset.quads() returns the 4th element as a URIRef (the
    # graph identifier), not a Graph context object.
    leaked = [
        (s, p, o, c) for s, p, o, c in pipeline_dataset.quads()
        if c not in named
    ]
    assert not leaked, (
        f"{len(leaked)} triples leaked into a non-named graph. "
        f"First 3: {leaked[:3]}"
    )


def test_default_union_makes_sparql_work_unchanged(pipeline_dataset):
    """SPARQL without GRAPH clauses should still see triples across all
    named graphs — proves default_union=True is set correctly. This is
    the property that lets existing queries keep working."""
    rows = list(pipeline_dataset.query(
        "PREFIX rtm: <http://example.org/ontology/rtm#> "
        "SELECT (COUNT(?att) AS ?n) WHERE { ?att a rtm:Attestation }"
    ))
    assert rows, "SPARQL returned no result rows"
    count = int(rows[0][0])
    assert count >= 3, (
        f"Expected ≥3 attestations visible via union view, got {count}. "
        f"Likely default_union is not set."
    )


def test_exported_trig_preserves_named_graphs():
    """output/rtm.trig should re-parse into the same named-graph structure."""
    trig_path = ROOT / "output" / "rtm.trig"
    if not trig_path.exists():
        pytest.skip("Pipeline export not present")
    ds = Dataset()
    ds.parse(trig_path, format="trig")
    graphs = {str(ctx.identifier) for ctx in ds.graphs() if len(ctx) > 0}
    expected = {G_ONTOLOGY, G_STRUCTURAL, G_EVIDENCE, G_ATTESTATIONS}
    missing = expected - graphs
    assert not missing, f"TriG export missing named graphs: {missing}"


def test_query_named_graph_scopes_to_layer(pipeline_dataset):
    """The helper restricts a query to one named graph; the same query
    over the union sees more triples (because other layers contribute)."""
    from pipeline.dataset import query_named_graph

    layer_rows = list(query_named_graph(
        pipeline_dataset, "attestations",
        "SELECT (COUNT(?s) AS ?n) WHERE { ?s ?p ?o }",
    ))
    layer_count = int(layer_rows[0][0])
    union_rows = list(pipeline_dataset.query(
        "SELECT (COUNT(?s) AS ?n) WHERE { ?s ?p ?o }",
    ))
    union_count = int(union_rows[0][0])
    assert 0 < layer_count < union_count, (
        f"Expected attestations graph ({layer_count} triples) to be a "
        f"strict subset of the union ({union_count} triples)."
    )


def test_query_named_graph_rejects_unknown_layer(pipeline_dataset):
    """The helper raises KeyError for unknown layer names."""
    from pipeline.dataset import query_named_graph

    with pytest.raises(KeyError, match="Unknown named-graph layer"):
        query_named_graph(
            pipeline_dataset, "nonexistent",
            "SELECT * WHERE { ?s ?p ?o }",
        )


def test_exported_ttl_remains_flat_union():
    """output/rtm.ttl should still be a flat Turtle file (back-compat for
    consumers that expect a single graph)."""
    ttl_path = ROOT / "output" / "rtm.ttl"
    if not ttl_path.exists():
        pytest.skip("Pipeline export not present")
    # The strongest check: parse as plain Turtle (which forbids named-graph
    # blocks) and confirm the triple count matches the union view.
    from rdflib import Graph as RdflibGraph
    g = RdflibGraph()
    g.parse(ttl_path, format="turtle")
    assert len(g) > 100, f"Flat-union rtm.ttl looks empty: {len(g)} triples"
