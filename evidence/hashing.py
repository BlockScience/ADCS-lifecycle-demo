"""Deterministic SHA-256 hashing for model identity and proof binding.

Pure functions — no state.

Evidence chain:
    structural graph (canonical N-Triples)
        -> hash_structural_model() -> model_hash
             -> hash_proof(script, model_hash) -> proof_hash
             -> hash_simulation(config, summary) -> sim_hash
                  -> hash_evidence(model_hash, proof_hash, sim_hash) -> evidence_hash

Reference: gds-proof/gds_proof/identity/hashing.py
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import sympy
from rdflib import Graph

if TYPE_CHECKING:
    from analysis.proof_scripts import ProofScript


def _serialize_for_hash(data: dict) -> str:
    """JSON-serialize a dict deterministically for hashing."""
    return json.dumps(data, sort_keys=True, default=str)


def hash_structural_model(graph: Graph) -> str:
    """Deterministic SHA-256 hash of the structural RDF model.

    Produces a canonical hash by:
    1. Extracting all triples as (s, p, o) N-Triples strings
    2. Replacing blank node identifiers with a content-based hash of
       all non-blank triples reachable from that blank node
    3. Sorting and hashing the result

    For simplicity, we flatten blank-node subgraphs: triples involving
    blank nodes are replaced by their grounded content (the non-blank
    subjects/objects they ultimately connect).
    """
    from rdflib import BNode, URIRef, Literal

    # Collect grounded triples (no blank nodes) directly
    grounded_lines: list[str] = []

    # For blank-node triples, collect the chain of properties
    # and serialize as: subject -> predicate chain -> leaf values
    def _nt_term(term):
        if isinstance(term, URIRef):
            return f"<{term}>"
        if isinstance(term, Literal):
            if term.datatype:
                return f'"{term}"^^<{term.datatype}>'
            return f'"{term}"'
        return f"_:blank"  # placeholder, will be skipped

    def _collect_bnode_properties(bnode, visited=None):
        """Recursively collect all property-value pairs from a blank node."""
        if visited is None:
            visited = set()
        if bnode in visited:
            return []
        visited.add(bnode)
        pairs = []
        for p, o in graph.predicate_objects(bnode):
            if isinstance(o, BNode):
                sub_pairs = _collect_bnode_properties(o, visited)
                for sp_, so_ in sub_pairs:
                    pairs.append((f"{_nt_term(p)}/{sp_}", so_))
            else:
                pairs.append((_nt_term(p), _nt_term(o)))
        return sorted(pairs)

    for s, p, o in graph:
        if isinstance(s, BNode) and isinstance(o, BNode):
            continue  # skip pure blank-to-blank (will be captured via parent)
        if isinstance(s, BNode):
            continue  # blank subjects are captured when their parent references them
        if isinstance(o, BNode):
            # Inline the blank node's content
            props = _collect_bnode_properties(o)
            for prop_path, value in props:
                grounded_lines.append(f"{_nt_term(s)} {_nt_term(p)}/{prop_path} {value} .")
        else:
            grounded_lines.append(f"{_nt_term(s)} {_nt_term(p)} {_nt_term(o)} .")

    canonical = "\n".join(sorted(grounded_lines))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_proof(script: ProofScript, model_hash: str) -> str:
    """Deterministic SHA-256 binding a proof script to a model version.

    Hashes lemma chain content together with model_hash.
    """
    lemma_records: list[dict[str, Any]] = []
    for lemma in script.lemmas:
        lemma_records.append(
            {
                "name": lemma.name,
                "kind": lemma.kind.value,
                "expr": sympy.srepr(lemma.expr),
                "expected": (
                    sympy.srepr(lemma.expected)
                    if lemma.expected is not None
                    else None
                ),
                "assumptions": lemma.assumptions,
                "depends_on": sorted(lemma.depends_on),
            }
        )
    data: dict[str, Any] = {
        "model_hash": model_hash,
        "target_invariant": script.target_invariant,
        "lemmas": lemma_records,
    }
    serialized = _serialize_for_hash(data)
    return hashlib.sha256(serialized.encode()).hexdigest()


def hash_simulation(
    sim_config: dict[str, Any],
    results_summary: dict[str, Any],
) -> str:
    """SHA-256 hash of simulation configuration + summary results."""
    data = {
        "config": sim_config,
        "results_summary": results_summary,
    }
    serialized = _serialize_for_hash(data)
    return hashlib.sha256(serialized.encode()).hexdigest()


def hash_evidence(
    model_hash: str,
    proof_hash: str | None = None,
    sim_hash: str | None = None,
) -> str:
    """Combined evidence hash from model, proof, and simulation hashes."""
    data: dict[str, str | None] = {
        "model_hash": model_hash,
        "proof_hash": proof_hash,
        "sim_hash": sim_hash,
    }
    serialized = _serialize_for_hash(data)
    return hashlib.sha256(serialized.encode()).hexdigest()
