"""Map closure-rule violations + reverification mismatches to pipeline stages.

Given the assembled RTM (a TriG file or a live Dataset) and a
VerificationReport, walks `prov:wasGeneratedBy` to stage-level
activities (p-plan:correspondsToStep) and returns the dedup'd
ordered set of pipeline stages that must re-run to restore closure.

Closes issue #3. Typer-based CLI (WP1 §4.6 discipline).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import typer
from rdflib import Dataset, URIRef


# Stage map kept in sync with PipelineState.activity_to_stage and
# traceability.plan_execution.STEP_NAMES; cross-checked by a unit test.
ACTIVITY_TO_STAGE: dict[str, int] = {
    "OntologyAssembly":    0,
    "LoadStructural":      1,
    "SymbolicAnalysis":    2,
    "NumericalSimulation": 3,
    "BindEvidence":        4,
    "AssembleRTM":         5,
    "Attest":              6,
    "ValidateShapes":      6,
    "AuditTrace":          7,
    "Report":              7,
    "Interrogate":         8,
}


@dataclass(frozen=True)
class StageRerun:
    stage_number: int
    step_name: str
    reason: str
    triggered_by: str


@dataclass
class RerunPlan:
    stages: list[StageRerun] = field(default_factory=list)
    structural_violations: list[dict] = field(default_factory=list)

    @property
    def stage_set(self) -> list[int]:
        return sorted({s.stage_number for s in self.stages})

    def to_dict(self) -> dict:
        return {
            "stages": [
                {"stage": s.stage_number, "step": s.step_name,
                 "reason": s.reason, "triggered_by": s.triggered_by}
                for s in self.stages
            ],
            "stage_set": self.stage_set,
            "structural_violations": self.structural_violations,
        }


_EVIDENCE_TO_STEP_Q = """
PREFIX prov:   <http://www.w3.org/ns/prov#>
PREFIX p-plan: <http://purl.org/net/p-plan#>
SELECT ?evidence ?activity ?step ?stepFragment WHERE {
    ?evidence prov:wasGeneratedBy ?activity .
    ?activity p-plan:correspondsToStep ?step .
    BIND(REPLACE(STR(?step), "^.*/", "") AS ?stepFragment)
    FILTER(?evidence = <%s>)
}
"""


def _stage_rerun_for_focus(ds: Dataset, focus_iri: str) -> StageRerun | None:
    """Find the pipeline stage that generated `focus_iri`, if any."""
    rows = list(ds.query(_EVIDENCE_TO_STEP_Q % focus_iri))
    if not rows:
        return None
    step_fragment = str(rows[0]["stepFragment"])
    stage_no = ACTIVITY_TO_STAGE.get(step_fragment)
    if stage_no is None:
        return None
    return StageRerun(
        stage_number=stage_no,
        step_name=step_fragment,
        reason=f"focus traces to {step_fragment}",
        triggered_by=focus_iri,
    )


def rerun_from_report(ds: Dataset, report) -> RerunPlan:
    """Build a RerunPlan from a VerificationReport.

    `report` is duck-typed: must expose `reverification_mismatches` and
    `shape_violations` iterables. Each reverification mismatch must
    have `.evidence`, `.expected`, `.actual` attributes; each shape
    violation must have `.shape`, `.focus`, `.message`, `.severity`.
    """
    plan = RerunPlan()
    seen: set[tuple[int, str]] = set()

    # Hash mismatches always trace to the producing stage. If the
    # evidence has no activity edge (shouldn't happen for well-formed
    # evidence) default to Stage 2 — symbolic analysis is the most
    # common producer.
    for m in report.reverification_mismatches:
        focus = m.evidence
        sr = _stage_rerun_for_focus(ds, focus) or StageRerun(
            stage_number=2,
            step_name="SymbolicAnalysis",
            reason="reverification mismatch (no activity edge)",
            triggered_by=focus,
        )
        sr = StageRerun(
            stage_number=sr.stage_number,
            step_name=sr.step_name,
            reason=(
                f"reverification mismatch: "
                f"expected {m.expected[:12]}, got {m.actual[:12]}"
            ),
            triggered_by=focus,
        )
        key = (sr.stage_number, sr.triggered_by)
        if key not in seen:
            plan.stages.append(sr)
            seen.add(key)

    # SHACL violations: if the focus node is producer-trackable (has a
    # generating activity), classify as a re-run; otherwise it's a
    # structural / human-side violation that no stage re-run can fix.
    for v in report.shape_violations:
        sr = _stage_rerun_for_focus(ds, v.focus)
        if sr is None:
            plan.structural_violations.append({
                "shape": v.shape, "focus": v.focus,
                "message": v.message, "severity": v.severity,
            })
            continue
        sr = StageRerun(
            stage_number=sr.stage_number,
            step_name=sr.step_name,
            reason=(
                f"SHACL {v.shape.rsplit('#', 1)[-1]}: "
                f"{v.message[:80]}"
            ),
            triggered_by=v.focus,
        )
        key = (sr.stage_number, sr.triggered_by)
        if key not in seen:
            plan.stages.append(sr)
            seen.add(key)

    plan.stages.sort(key=lambda s: (s.stage_number, s.step_name))
    return plan


def rerun_from_dataset(ds: Dataset, *, requirement: str | None = None) -> RerunPlan:
    """Run verification on `ds` and convert the report to a RerunPlan."""
    from traceability.verification import verify
    report = verify(ds, skip_reverification=False)
    plan = rerun_from_report(ds, report)
    if requirement:
        plan = _filter_by_requirement(ds, plan, requirement)
    return plan


def _filter_by_requirement(ds: Dataset, plan: RerunPlan, req_name: str) -> RerunPlan:
    """Restrict the plan to focus nodes reachable from the named requirement."""
    q = """
    PREFIX rtm:   <http://example.org/ontology/rtm#>
    PREFIX sysml: <https://www.omg.org/spec/SysML/2.0/>
    SELECT ?node WHERE {
        ?req sysml:declaredName "%s" .
        { ?node rtm:addresses ?req . }
        UNION { ?att rtm:attests ?req ; rtm:hasEvidence ?node . }
    }
    """ % req_name
    reachable = {str(r["node"]) for r in ds.query(q)}
    return RerunPlan(
        stages=[s for s in plan.stages if s.triggered_by in reachable],
        structural_violations=[
            v for v in plan.structural_violations if v["focus"] in reachable
        ],
    )


def render_plan(plan: RerunPlan, fmt: str = "md") -> str:
    """Render a RerunPlan as Markdown / JSON / plain text."""
    if fmt == "json":
        return json.dumps(plan.to_dict(), indent=2)
    if fmt == "md":
        lines = ["# Pipeline rerun plan", ""]
        if not plan.stages:
            lines.append("- No stages require re-running.")
        else:
            lines.append(f"Stages to re-run (in order): {plan.stage_set}")
            for s in plan.stages:
                lines.append(f"- **Stage {s.stage_number} ({s.step_name})** — {s.reason}")
        if plan.structural_violations:
            lines.append("")
            lines.append("## Structural violations (no rerun remedy)")
            for v in plan.structural_violations:
                shape = v['shape'].rsplit('#', 1)[-1]
                lines.append(f"- {shape}: {v['message'][:120]}")
        return "\n".join(lines)
    # txt
    if plan.stages:
        lines = [f"Rerun stages: {plan.stage_set}"]
    else:
        lines = ["No stages require re-running."]
    for s in plan.stages:
        lines.append(f"  [Stage {s.stage_number}] {s.step_name}: {s.reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    md = "md"
    json = "json"
    txt = "txt"


app = typer.Typer(
    add_completion=False,
    help=(
        "Map closure-rule violations + hash mismatches to pipeline "
        "stages (issue #3)."
    ),
    no_args_is_help=False,
)


@app.command()
def cli(
    input: Annotated[Path, typer.Option(
        "--input", help="TriG dataset to analyse.",
    )] = Path("output/rtm.trig"),
    requirement: Annotated[str | None, typer.Option(
        "--requirement",
        help="Restrict the rerun plan to evidence reachable from this requirement (e.g. REQ-003).",
    )] = None,
    format: Annotated[OutputFormat, typer.Option(
        "--format", "-f", help="Output format.",
    )] = OutputFormat.md,
) -> None:
    """Print the pipeline stages that must re-run to restore closure."""
    if not input.exists():
        typer.echo(f"Input not found: {input}.", err=True)
        raise typer.Exit(code=2)
    ds = Dataset(default_union=True)
    ds.parse(input, format="trig")
    plan = rerun_from_dataset(ds, requirement=requirement)
    typer.echo(render_plan(plan, fmt=format.value))
    if plan.stages or plan.structural_violations:
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
