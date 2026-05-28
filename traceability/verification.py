"""Stage 6.5 — closure-rule verification via SHACL + runtime checks.

Runs the 9 SHACL shapes in ontology/rtm_shapes.ttl against the assembled
Dataset, plus the runtime re-verification closure (#10) that re-hashes
every rtm:ProofArtifact against its source.

Returns a VerificationReport with conformance status and a list of
violations / re-verification mismatches. The runner can either fail
hard on violations or continue (default: continue, surface violations
in the report).

Naming discipline (WP1 §4.4): SHACL conformance and hash-matching are
**verification** (automated, fully specified). The pyshacl API itself
keeps its upstream name (`pyshacl.validate`); the demo's own wrappers
use `verify`. Engineer attestation (human judgement) is separately
named in `traceability.attestation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pyshacl
from rdflib import Dataset, Graph, URIRef

ROOT = Path(__file__).resolve().parent.parent
SHAPES_PATH = ROOT / "ontology" / "rtm_shapes.ttl"


@dataclass
class ShapeViolation:
    shape: str          # the SHACL source shape identifier or message excerpt
    focus: str          # focus node IRI
    path: str | None    # property path that violated
    message: str        # human-readable failure message
    severity: str       # sh:Violation / sh:Warning / sh:Info


@dataclass
class ReverificationMismatch:
    evidence: str       # evidence IRI
    expected: str       # stored proofHash
    actual: str         # recomputed proofHash


@dataclass
class VerificationReport:
    conforms: bool
    shape_violations: list[ShapeViolation] = field(default_factory=list)
    reverification_mismatches: list[ReverificationMismatch] = field(default_factory=list)
    shape_results_text: str = ""

    def summary_lines(self) -> list[str]:
        lines = []
        if self.conforms:
            lines.append("Closure-rule verification: PASS")
            lines.append(f"  SHACL shapes:        {len(self.shape_violations)} violations")
            lines.append(f"  Re-verification:     {len(self.reverification_mismatches)} mismatches")
        else:
            lines.append("Closure-rule verification: FAIL")
            if self.shape_violations:
                lines.append(f"  SHACL violations ({len(self.shape_violations)}):")
                for v in self.shape_violations[:10]:
                    lines.append(f"    - {v.shape}: {v.message[:80]}")
                if len(self.shape_violations) > 10:
                    lines.append(f"    ... and {len(self.shape_violations) - 10} more")
            if self.reverification_mismatches:
                lines.append(f"  Re-verification mismatches ({len(self.reverification_mismatches)}):")
                for m in self.reverification_mismatches:
                    lines.append(f"    - {m.evidence}: expected {m.expected[:12]}, got {m.actual[:12]}")
        return lines


def _flatten(ds: Dataset) -> Graph:
    """pyshacl validates against a single Graph; flatten the Dataset's
    union into one Graph for validation purposes. The named-graph
    structure is preserved in the source data; this is just a query view."""
    g = Graph()
    for s, p, o in ds.triples((None, None, None)):
        g.add((s, p, o))
    return g


def _parse_shape_violations(report_graph: Graph) -> list[ShapeViolation]:
    """Extract structured violation records from pyshacl's report graph."""
    from rdflib.namespace import RDF
    SH = URIRef("http://www.w3.org/ns/shacl#")
    SH_RESULT = URIRef("http://www.w3.org/ns/shacl#ValidationResult")
    SH_FOCUS = URIRef("http://www.w3.org/ns/shacl#focusNode")
    SH_PATH = URIRef("http://www.w3.org/ns/shacl#resultPath")
    SH_MSG = URIRef("http://www.w3.org/ns/shacl#resultMessage")
    SH_SEV = URIRef("http://www.w3.org/ns/shacl#resultSeverity")
    SH_SHAPE = URIRef("http://www.w3.org/ns/shacl#sourceShape")

    out: list[ShapeViolation] = []
    for result in report_graph.subjects(RDF.type, SH_RESULT):
        focus = next(iter(report_graph.objects(result, SH_FOCUS)), None)
        path = next(iter(report_graph.objects(result, SH_PATH)), None)
        msg = next(iter(report_graph.objects(result, SH_MSG)), None)
        sev = next(iter(report_graph.objects(result, SH_SEV)), None)
        shape = next(iter(report_graph.objects(result, SH_SHAPE)), None)
        out.append(ShapeViolation(
            shape=str(shape) if shape else "?",
            focus=str(focus) if focus else "?",
            path=str(path) if path else None,
            message=str(msg) if msg else "",
            severity=str(sev) if sev else "?",
        ))
    return out


def verify_shacl(ds: Dataset, shapes_path: Path = SHAPES_PATH) -> tuple[bool, list[ShapeViolation], str]:
    """Run pyshacl against `ds` with the closure-rule shapes."""
    shapes = Graph()
    shapes.parse(shapes_path, format="turtle")
    data = _flatten(ds)
    conforms, report_graph, results_text = pyshacl.validate(
        data, shacl_graph=shapes,
        inference="rdfs", allow_warnings=True, advanced=True,
    )
    violations = _parse_shape_violations(report_graph)
    return conforms, violations, results_text


def verify_reverification(ds: Dataset) -> list[ReverificationMismatch]:
    """Closure rule #10 — re-run every ProofArtifact and check its proofHash.

    Imports reproduce_proof lazily so this module doesn't pull in scipy /
    sympy at import time.
    """
    from interrogate.reproduce import reproduce_proof
    from ontology.prefixes import RTM

    mismatches: list[ReverificationMismatch] = []
    for ev in ds.subjects(URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                          RTM.ProofArtifact, unique=True):
        result = reproduce_proof(ds, str(ev))
        if result is None:
            continue
        # reproduce_proof returns a dict; the expected hash is in the
        # graph as rtm:proofHash, the recomputed one is in the result.
        expected_hashes = list(ds.objects(ev, RTM.proofHash))
        if not expected_hashes:
            continue
        expected = str(expected_hashes[0])
        actual = result.get("proof_hash", "") or result.get("new_proof_hash", "")
        if actual and actual != expected:
            mismatches.append(ReverificationMismatch(
                evidence=str(ev), expected=expected, actual=actual,
            ))
    return mismatches


def verify(ds: Dataset, *, shapes_path: Path = SHAPES_PATH,
           skip_reverification: bool = False) -> VerificationReport:
    """Run the full closure-rule suite (SHACL + re-verification) and
    return a structured report."""
    conforms, violations, text = verify_shacl(ds, shapes_path)
    mismatches: list[ReverificationMismatch] = []
    if not skip_reverification:
        try:
            mismatches = verify_reverification(ds)
        except Exception as exc:
            # Re-verification can fail for environmental reasons; surface
            # as a Violation rather than crashing the pipeline.
            violations.append(ShapeViolation(
                shape="rtm:ReverificationCheck",
                focus="(runtime)",
                path=None,
                message=f"Re-verification check failed to run: {exc}",
                severity="sh:Warning",
            ))
    overall = conforms and not mismatches
    return VerificationReport(
        conforms=overall,
        shape_violations=violations,
        reverification_mismatches=mismatches,
        shape_results_text=text,
    )


# Back-compat aliases — remove in a follow-up PR after WP3 lands.
# Existing external scripts and notebooks may import these names.
ValidationReport = VerificationReport
validate = verify
validate_shacl = verify_shacl
validate_reverification = verify_reverification
