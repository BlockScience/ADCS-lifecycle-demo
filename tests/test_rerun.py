"""Tests for interrogate.rerun — issue #3 acceptance criteria.

Coverage map (issue #3 ACs):
  AC #1: closed/valid RTM -> empty stage set
  AC #2: hash mismatch on a ProofArtifact -> Stage 2 returned
  AC #3: hash mismatch on a SimulationResult -> Stage 3 returned
  AC #4: model perturbation invalidating multiple artifacts -> union, ordered
  AC #5: SHACL violation that doesn't trace to evidence -> reported separately
  AC #6: CLI documented in README (asserted by tests/test_cli.py)
  AC #7: Stage 6.5 banner extended to list re-run stages on violations
         (asserted by visual inspection in run_stage_6_5_verify_closure;
          unit-tested here by ensuring rerun_from_report integrates cleanly)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from rdflib import Dataset, URIRef

from interrogate.rerun import (
    ACTIVITY_TO_STAGE,
    RerunPlan,
    rerun_from_dataset,
    rerun_from_report,
    render_plan,
)
from ontology.prefixes import RTM
from pipeline.runner import run_pipeline
from traceability.verification import (
    ReverificationMismatch,
    ShapeViolation,
    VerificationReport,
    verify,
)


@pytest.fixture(scope="module")
def closed_rtm() -> Dataset:
    """A fully attested pipeline run — should satisfy every shape."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return run_pipeline(auto_attest=True)


# ---------------------------------------------------------------------------
# AC #1: closed/valid RTM -> empty stage set (no false positives)
# ---------------------------------------------------------------------------

def test_ac1_closed_rtm_empty_stage_set(closed_rtm):
    report = verify(closed_rtm, skip_reverification=False)
    plan = rerun_from_report(closed_rtm, report)
    assert plan.stage_set == [], (
        f"Expected no re-run stages on a closed RTM, got {plan.stage_set}. "
        f"Reasons: {[s.reason for s in plan.stages]}"
    )


# ---------------------------------------------------------------------------
# AC #2: hash mismatch on a ProofArtifact -> Stage 2 (symbolic) returned
# ---------------------------------------------------------------------------

def test_ac2_proof_hash_mismatch_returns_stage_2(closed_rtm):
    """Synthetic ReverificationMismatch on a real ProofArtifact must
    route to Stage 2 because that's where symbolic analysis runs."""
    type_iri = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    ev_iris = list(closed_rtm.subjects(type_iri, RTM.ProofArtifact, unique=True))
    assert ev_iris, "Expected at least one rtm:ProofArtifact in the closed RTM"

    report = VerificationReport(conforms=False, reverification_mismatches=[
        ReverificationMismatch(
            evidence=str(ev_iris[0]),
            expected="a" * 64,
            actual="b" * 64,
        ),
    ])
    plan = rerun_from_report(closed_rtm, report)
    assert 2 in plan.stage_set, (
        f"Expected Stage 2 in plan; got {plan.stage_set}"
    )


# ---------------------------------------------------------------------------
# AC #3: hash mismatch on a SimulationResult -> Stage 3 returned
# ---------------------------------------------------------------------------

def test_ac3_sim_violation_returns_stage_3(closed_rtm):
    """A shape violation focused on a SimulationResult must route to Stage 3."""
    type_iri = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    sim_iris = list(closed_rtm.subjects(type_iri, RTM.SimulationResult, unique=True))
    assert sim_iris, "Expected at least one rtm:SimulationResult in the closed RTM"

    report = VerificationReport(conforms=False, shape_violations=[
        ShapeViolation(
            shape="rtm:EvidenceShape",
            focus=str(sim_iris[0]),
            path=None,
            message="(synthetic violation for test)",
            severity="sh:Violation",
        ),
    ])
    plan = rerun_from_report(closed_rtm, report)
    assert 3 in plan.stage_set, (
        f"Expected Stage 3 in plan; got {plan.stage_set}"
    )


# ---------------------------------------------------------------------------
# AC #4: multiple invalidated artifacts -> union of stages, ordered
# ---------------------------------------------------------------------------

def test_ac4_multiple_invalidations_ordered_union(closed_rtm):
    type_iri = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    proof = next(iter(closed_rtm.subjects(type_iri, RTM.ProofArtifact, unique=True)))
    sim = next(iter(closed_rtm.subjects(type_iri, RTM.SimulationResult, unique=True)))

    report = VerificationReport(
        conforms=False,
        reverification_mismatches=[ReverificationMismatch(
            evidence=str(proof), expected="a", actual="b",
        )],
        shape_violations=[ShapeViolation(
            shape="rtm:EvidenceShape", focus=str(sim),
            path=None, message="x", severity="sh:Violation",
        )],
    )
    plan = rerun_from_report(closed_rtm, report)
    assert plan.stage_set == [2, 3], (
        f"Expected [2, 3] in pipeline order; got {plan.stage_set}"
    )


# ---------------------------------------------------------------------------
# AC #5: attestation-level SHACL violation -> structural_violations only
# ---------------------------------------------------------------------------

def test_ac5_structural_violation_reported_separately(closed_rtm):
    """Attestations are not stage-producible; a SHACL violation on an
    attestation node should appear in structural_violations, not in
    stages."""
    type_iri = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    att_iris = list(closed_rtm.subjects(type_iri, RTM.Attestation, unique=True))
    assert att_iris, "Expected at least one rtm:Attestation"

    report = VerificationReport(conforms=False, shape_violations=[
        ShapeViolation(
            shape="rtm:AttestationShape",
            focus=str(att_iris[0]),
            path=None,
            message="missing adequacy assumption",
            severity="sh:Violation",
        ),
    ])
    plan = rerun_from_report(closed_rtm, report)
    assert plan.stage_set == [], (
        f"Attestation violations should not trigger stage re-runs; "
        f"got {plan.stage_set}"
    )
    assert len(plan.structural_violations) == 1


# ---------------------------------------------------------------------------
# AC #7 + plumbing: stage map covers every known step
# ---------------------------------------------------------------------------

def test_activity_to_stage_table_covers_known_steps():
    """ACTIVITY_TO_STAGE in rerun.py must stay in sync with STEP_NAMES."""
    from traceability.plan_execution import STEP_NAMES
    missing = STEP_NAMES - set(ACTIVITY_TO_STAGE)
    assert not missing, f"Step names without stage map: {missing}"


def test_render_plan_md_includes_stage_set_when_nonempty(closed_rtm):
    """Markdown rendering surfaces the stage_set on a non-empty plan."""
    type_iri = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    proof = next(iter(closed_rtm.subjects(type_iri, RTM.ProofArtifact, unique=True)))
    report = VerificationReport(conforms=False, reverification_mismatches=[
        ReverificationMismatch(evidence=str(proof), expected="a", actual="b"),
    ])
    plan = rerun_from_report(closed_rtm, report)
    md = render_plan(plan, fmt="md")
    assert "Stages to re-run" in md
    assert "Stage 2" in md


def test_render_plan_empty_says_no_rerun_needed(closed_rtm):
    report = verify(closed_rtm, skip_reverification=False)
    plan = rerun_from_report(closed_rtm, report)
    md = render_plan(plan, fmt="md")
    assert "No stages require re-running." in md


def test_render_plan_json_round_trips(closed_rtm):
    """JSON render is parseable + contains the stage_set key."""
    import json as _json
    type_iri = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    proof = next(iter(closed_rtm.subjects(type_iri, RTM.ProofArtifact, unique=True)))
    report = VerificationReport(conforms=False, reverification_mismatches=[
        ReverificationMismatch(evidence=str(proof), expected="a", actual="b"),
    ])
    plan = rerun_from_report(closed_rtm, report)
    js = render_plan(plan, fmt="json")
    parsed = _json.loads(js)
    assert "stage_set" in parsed
    assert 2 in parsed["stage_set"]
