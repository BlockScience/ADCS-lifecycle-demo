"""Forward / backward / bidirectional traceability audit.

Computes the trace-matrix surface Jama and OSLC-compliant tools expose,
plus orphan detection. The key design property is that forward and
backward checks are *independent*: they run separately and emit
separate failure lists, so error messages identify which direction
broke and why. Bidirectional traceability is a derived predicate
(`forward.passed ∧ backward.passed`), never a primary check.

Three failure modes the audit distinguishes:
  - Forward fail, backward pass: requirement R isn't reachable to an
    attested evidence link. The structural side is intact; evidence
    generation missed this requirement.
  - Backward fail, forward pass: attestation A references evidence E
    that doesn't declare rtm:addresses on R. The attestation produces
    claims unsupported by structural intent.
  - Both fail: both message sets are reported, named separately.

Audit output is rendered as CSV / Markdown / JSON-LD AND emitted as
RDF triples into the <adcs:audit> named graph so the audit is itself
queryable and traceable.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from rdflib import Dataset, Literal as RdfLiteral, URIRef
from rdflib.namespace import RDF, XSD

from ontology.prefixes import ADCS, DCTERMS, EARL, G_AUDIT, PROV, RTM

Direction = Literal["forward", "backward", "bidirectional"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Failure:
    """One trace failure with a precise reason."""
    subject: str       # the IRI / name being checked (requirement or attestation)
    reason: str        # human-readable explanation
    details: dict = field(default_factory=dict)


@dataclass
class DirectionResult:
    direction: str     # "forward" or "backward"
    passed: bool
    checked_count: int
    failures: list[Failure] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (f"{self.direction.title():<10} {status:<5} "
                f"({self.checked_count} checked, {len(self.failures)} failures)")


@dataclass
class BidirectionalResult:
    forward: DirectionResult
    backward: DirectionResult

    @property
    def passed(self) -> bool:
        return self.forward.passed and self.backward.passed

    def summary(self) -> str:
        lines = []
        if self.passed:
            lines.append("Bidirectional traceability: PASS")
        else:
            broken = []
            if not self.forward.passed:
                broken.append("forward")
            if not self.backward.passed:
                broken.append("backward")
            lines.append(f"Bidirectional traceability: FAIL "
                         f"(broken direction(s): {', '.join(broken)})")
        lines.append(f"  {self.forward.summary()}")
        lines.append(f"  {self.backward.summary()}")
        return "\n".join(lines)


@dataclass
class CoverageCell:
    requirement: str
    evidence: str
    status: str        # "covered+passed" | "covered+failed" | "covered+cantTell" | "uncovered"


@dataclass
class OrphanReport:
    requirements_without_evidence: list[str] = field(default_factory=list)
    evidence_without_requirement: list[str] = field(default_factory=list)
    attestations_with_broken_refs: list[str] = field(default_factory=list)

    @property
    def any(self) -> bool:
        return bool(
            self.requirements_without_evidence
            or self.evidence_without_requirement
            or self.attestations_with_broken_refs
        )


@dataclass
class AuditReport:
    forward: DirectionResult
    backward: DirectionResult
    coverage: list[CoverageCell]
    orphans: OrphanReport
    timestamp: str

    @property
    def passed(self) -> bool:
        return (
            self.forward.passed
            and self.backward.passed
            and not self.orphans.any
        )

    def bidirectional(self) -> BidirectionalResult:
        return BidirectionalResult(forward=self.forward, backward=self.backward)


# ---------------------------------------------------------------------------
# SPARQL queries
# ---------------------------------------------------------------------------

_ADCS_REQUIREMENTS_Q = """
PREFIX sysml: <https://www.omg.org/spec/SysML/2.0/>
SELECT ?req ?name WHERE {
    ?req a sysml:RequirementDefinition ;
         sysml:declaredName ?name .
    FILTER(STRSTARTS(?name, "REQ-"))
}
ORDER BY ?name
"""

_FORWARD_TRACE_Q = """
PREFIX rtm:   <http://example.org/ontology/rtm#>
PREFIX sysml: <https://www.omg.org/spec/SysML/2.0/>
PREFIX earl:  <http://www.w3.org/ns/earl#>
SELECT ?req ?name ?ev ?att ?outcome WHERE {
    ?req a sysml:RequirementDefinition ;
         sysml:declaredName ?name .
    FILTER(STRSTARTS(?name, "REQ-"))
    OPTIONAL {
        ?ev rtm:addresses ?req .
        OPTIONAL {
            ?att rtm:attests ?req ;
                 rtm:hasEvidence ?ev ;
                 rtm:hasOutcome ?outcome .
        }
    }
}
"""

_BACKWARD_TRACE_Q = """
PREFIX rtm: <http://example.org/ontology/rtm#>
SELECT ?att ?req ?ev WHERE {
    ?att a rtm:Attestation ;
         rtm:attests ?req ;
         rtm:hasEvidence ?ev .
}
"""


# ---------------------------------------------------------------------------
# Direction checks — INDEPENDENT
# ---------------------------------------------------------------------------
# NOTE: every ds.query(...) below is INTENTIONALLY UNION-SCOPED — the
# audit joins across <adcs:structural>, <adcs:evidence>, and
# <adcs:attestations>, relying on Dataset(default_union=True) so the
# queries omit GRAPH clauses. For single-layer queries use
# pipeline.dataset.query_named_graph instead.

def forward_trace(ds: Dataset) -> DirectionResult:
    """For every ADCS requirement, confirm there exists at least one
    addressing evidence artifact AND at least one attestation that
    attests this requirement with an outcome value. Outcome value is
    informational — declined attestations (earl:failed / earl:cantTell)
    still count as forward-reachable; the failure mode is "no path at all."
    """
    reqs: dict[str, dict] = {}
    for row in ds.query(_FORWARD_TRACE_Q):
        name = str(row["name"])
        e = reqs.setdefault(name, {"req": str(row["req"]), "evs": set(), "atts": set()})
        if row["ev"]:
            e["evs"].add(str(row["ev"]))
        if row["att"]:
            e["atts"].add(str(row["att"]))

    failures: list[Failure] = []
    for name, info in reqs.items():
        if not info["evs"]:
            failures.append(Failure(
                subject=name,
                reason=f"requirement {name} is not reached by any rtm:addresses link",
                details={"evidence_count": 0, "attestation_count": len(info["atts"])},
            ))
        elif not info["atts"]:
            failures.append(Failure(
                subject=name,
                reason=f"requirement {name} has evidence but no attestation",
                details={"evidence_count": len(info["evs"]), "attestation_count": 0},
            ))

    return DirectionResult(
        direction="forward",
        passed=not failures,
        checked_count=len(reqs),
        failures=failures,
    )


def backward_trace(ds: Dataset) -> DirectionResult:
    """For every attestation, confirm every linked evidence artifact
    declares rtm:addresses on the attested requirement. Catches the
    case where attestation produces claims unsupported by structural
    intent."""
    failures: list[Failure] = []
    seen_attestations: set[str] = set()
    for row in ds.query(_BACKWARD_TRACE_Q):
        att, req, ev = str(row["att"]), str(row["req"]), str(row["ev"])
        seen_attestations.add(att)
        # Check ev rtm:addresses req exists.
        addresses_ok = (URIRef(ev), RTM.addresses, URIRef(req)) in ds
        if not addresses_ok:
            failures.append(Failure(
                subject=att,
                reason=(
                    f"attestation references evidence {ev.rsplit('/', 1)[-1]} "
                    f"that does not declare rtm:addresses on {req.rsplit('/', 1)[-1]}"
                ),
                details={"attestation": att, "evidence": ev, "requirement": req},
            ))

    return DirectionResult(
        direction="backward",
        passed=not failures,
        checked_count=len(seen_attestations),
        failures=failures,
    )


def bidirectional_trace(ds: Dataset) -> BidirectionalResult:
    """Conjunction of forward and backward. Each direction's failures
    are preserved in the result so error messages name which direction
    (or both) broke."""
    return BidirectionalResult(
        forward=forward_trace(ds),
        backward=backward_trace(ds),
    )


# ---------------------------------------------------------------------------
# Coverage matrix + orphans
# ---------------------------------------------------------------------------

def coverage_matrix(ds: Dataset) -> list[CoverageCell]:
    """One row per (requirement, addressing-evidence) pair, with an
    outcome-derived status. Uncovered requirements appear with
    evidence='-' and status='uncovered'."""
    cells: list[CoverageCell] = []
    seen_with_evidence: set[str] = set()
    for row in ds.query(_FORWARD_TRACE_Q):
        name = str(row["name"])
        ev = row["ev"]
        outcome = row["outcome"]
        if ev:
            ev_name = str(ev).rsplit("/", 1)[-1]
            seen_with_evidence.add(name)
            if outcome == EARL.passed:
                status = "covered+passed"
            elif outcome == EARL.failed:
                status = "covered+failed"
            elif outcome == EARL.cantTell:
                status = "covered+cantTell"
            elif outcome is None:
                status = "covered+unattested"
            else:
                status = f"covered+{str(outcome).rsplit('#', 1)[-1]}"
            cells.append(CoverageCell(requirement=name, evidence=ev_name, status=status))

    # Uncovered requirements
    for row in ds.query(_ADCS_REQUIREMENTS_Q):
        name = str(row["name"])
        if name not in seen_with_evidence:
            cells.append(CoverageCell(requirement=name, evidence="-", status="uncovered"))

    cells.sort(key=lambda c: (c.requirement, c.evidence))
    return cells


def orphans(ds: Dataset) -> OrphanReport:
    """Find requirements with no evidence, evidence with no requirement,
    and attestations with references that don't resolve."""
    report = OrphanReport()

    # Requirements with no evidence
    q_req_no_ev = """
    PREFIX rtm: <http://example.org/ontology/rtm#>
    PREFIX sysml: <https://www.omg.org/spec/SysML/2.0/>
    SELECT ?name WHERE {
        ?req a sysml:RequirementDefinition ; sysml:declaredName ?name .
        FILTER(STRSTARTS(?name, "REQ-"))
        FILTER NOT EXISTS { ?ev rtm:addresses ?req }
    }
    """
    for row in ds.query(q_req_no_ev):
        report.requirements_without_evidence.append(str(row["name"]))

    # Evidence with no requirement
    q_ev_no_req = """
    PREFIX rtm: <http://example.org/ontology/rtm#>
    SELECT ?ev WHERE {
        ?ev a ?type .
        FILTER(?type IN (rtm:ProofArtifact, rtm:SimulationResult))
        FILTER NOT EXISTS { ?ev rtm:addresses ?req }
    }
    """
    for row in ds.query(q_ev_no_req):
        report.evidence_without_requirement.append(str(row["ev"]))

    # Attestations referencing nonexistent requirements
    q_broken_att = """
    PREFIX rtm: <http://example.org/ontology/rtm#>
    PREFIX sysml: <https://www.omg.org/spec/SysML/2.0/>
    SELECT ?att WHERE {
        ?att a rtm:Attestation ; rtm:attests ?req .
        FILTER NOT EXISTS { ?req a sysml:RequirementDefinition }
    }
    """
    for row in ds.query(q_broken_att):
        report.attestations_with_broken_refs.append(str(row["att"]))

    return report


# ---------------------------------------------------------------------------
# Full audit
# ---------------------------------------------------------------------------

def audit(ds: Dataset) -> AuditReport:
    """Run the full audit suite. Forward and backward are independent;
    bidirectional is derived via AuditReport.bidirectional()."""
    return AuditReport(
        forward=forward_trace(ds),
        backward=backward_trace(ds),
        coverage=coverage_matrix(ds),
        orphans=orphans(ds),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _render_markdown(report: AuditReport) -> str:
    lines = ["# RTM Traceability Audit", f"_generated {report.timestamp}_", ""]
    lines.append("## Direction summary")
    lines.append(f"- {report.forward.summary()}")
    lines.append(f"- {report.backward.summary()}")
    bidirect = report.bidirectional()
    lines.append(f"- **Bidirectional: {'PASS' if bidirect.passed else 'FAIL'}**")
    lines.append("")

    for direction in (report.forward, report.backward):
        if direction.failures:
            lines.append(f"## {direction.direction.title()} failures ({len(direction.failures)})")
            for f in direction.failures:
                lines.append(f"- **{f.subject}**: {f.reason}")
            lines.append("")

    lines.append("## Coverage matrix")
    lines.append("| Requirement | Evidence | Status |")
    lines.append("| --- | --- | --- |")
    for cell in report.coverage:
        lines.append(f"| {cell.requirement} | {cell.evidence} | {cell.status} |")
    lines.append("")

    lines.append("## Orphans")
    if not report.orphans.any:
        lines.append("None.")
    else:
        if report.orphans.requirements_without_evidence:
            lines.append("**Requirements without evidence:**")
            for name in report.orphans.requirements_without_evidence:
                lines.append(f"- {name}")
        if report.orphans.evidence_without_requirement:
            lines.append("**Evidence without requirement:**")
            for iri in report.orphans.evidence_without_requirement:
                lines.append(f"- {iri}")
        if report.orphans.attestations_with_broken_refs:
            lines.append("**Attestations with broken references:**")
            for iri in report.orphans.attestations_with_broken_refs:
                lines.append(f"- {iri}")
    return "\n".join(lines)


def _render_csv(report: AuditReport) -> str:
    """Coverage matrix as CSV. Other report sections are not naturally
    tabular; use Markdown / JSON for those."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["requirement", "evidence", "status"])
    for cell in report.coverage:
        w.writerow([cell.requirement, cell.evidence, cell.status])
    return out.getvalue()


def _render_json(report: AuditReport) -> str:
    return json.dumps({
        "timestamp": report.timestamp,
        "passed": report.passed,
        "forward": {
            "passed": report.forward.passed,
            "checked_count": report.forward.checked_count,
            "failures": [asdict(f) for f in report.forward.failures],
        },
        "backward": {
            "passed": report.backward.passed,
            "checked_count": report.backward.checked_count,
            "failures": [asdict(f) for f in report.backward.failures],
        },
        "bidirectional": {
            "passed": report.bidirectional().passed,
        },
        "coverage": [asdict(c) for c in report.coverage],
        "orphans": asdict(report.orphans),
    }, indent=2)


def render_report(report: AuditReport, fmt: Literal["csv", "md", "json"] = "md") -> str:
    if fmt == "csv":
        return _render_csv(report)
    if fmt == "json":
        return _render_json(report)
    return _render_markdown(report)


# ---------------------------------------------------------------------------
# RDF emission of the audit summary into <adcs:audit>
# ---------------------------------------------------------------------------

def emit_audit_graph(ds: Dataset, report: AuditReport) -> URIRef:
    """Write the audit summary as RDF triples into <adcs:audit> so the
    audit itself is queryable and traceable. Returns the audit-report
    resource IRI."""
    audit_g = ds.graph(URIRef(G_AUDIT))
    audit_iri = ADCS[f"audit/report-{report.timestamp.replace(':', '').replace('-', '').replace('.', '')[:18]}"]

    audit_g.add((audit_iri, RDF.type, RTM.AuditReport)) if hasattr(RTM, "AuditReport") else None
    audit_g.add((audit_iri, DCTERMS.created,
                 RdfLiteral(report.timestamp, datatype=XSD.dateTime)))
    audit_g.add((audit_iri, RTM.forwardPassed,
                 RdfLiteral(report.forward.passed, datatype=XSD.boolean)))
    audit_g.add((audit_iri, RTM.backwardPassed,
                 RdfLiteral(report.backward.passed, datatype=XSD.boolean)))
    audit_g.add((audit_iri, RTM.bidirectionalPassed,
                 RdfLiteral(report.bidirectional().passed, datatype=XSD.boolean)))
    audit_g.add((audit_iri, RTM.forwardFailures,
                 RdfLiteral(len(report.forward.failures), datatype=XSD.integer)))
    audit_g.add((audit_iri, RTM.backwardFailures,
                 RdfLiteral(len(report.backward.failures), datatype=XSD.integer)))
    audit_g.add((audit_iri, PROV.wasGeneratedBy,
                 ADCS["agent/audit-module"]))
    return audit_iri


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> int:
    """Standalone CLI: load output/rtm.trig (or a path given via --input)
    and run an audit in the requested direction / format."""
    import argparse

    parser = argparse.ArgumentParser(description="RTM Traceability Audit")
    parser.add_argument("--direction",
                        choices=["forward", "backward", "bidirectional", "full"],
                        default="full",
                        help="Which check(s) to run (default: full audit)")
    parser.add_argument("--format", choices=["csv", "md", "json"], default="md",
                        help="Report format")
    parser.add_argument("--input", default="output/rtm.trig",
                        help="Path to the .trig file to audit (default: output/rtm.trig)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}. Run the pipeline first.", file=sys.stderr)
        return 2

    ds = Dataset(default_union=True)
    ds.parse(input_path, format="trig")

    if args.direction == "forward":
        result = forward_trace(ds)
        print(result.summary())
        for f in result.failures:
            print(f"  - {f.subject}: {f.reason}")
        return 0 if result.passed else 1

    if args.direction == "backward":
        result = backward_trace(ds)
        print(result.summary())
        for f in result.failures:
            print(f"  - {f.subject}: {f.reason}")
        return 0 if result.passed else 1

    if args.direction == "bidirectional":
        result = bidirectional_trace(ds)
        print(result.summary())
        return 0 if result.passed else 1

    # Full audit
    report = audit(ds)
    print(render_report(report, fmt=args.format))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(_cli())
