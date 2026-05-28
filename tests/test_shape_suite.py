"""Phase G — closure-rule suite tests.

For each of the 9 SHACL shapes plus the runtime re-verification check,
we have:
  - a positive case: the nominal pipeline graph satisfies the shape
  - a negative case: a deliberately broken graph fails with the expected
    shape's message substring

The 9 SHACL shapes (per /Users/z/.claude/plans/i-want-to-look-hidden-balloon.md):
  #1  rtm:AttestationShape
  #2  rtm:PlanInstantiationShape
  #3  rtm:EvidenceShape / ProofArtifactShape
  #4  rtm:RequirementShape
  #5  rtm:GsnArgumentShape (StrategySupported + GoalSupported + GsnTextShape)
  #6  rtm:ProvenanceShape (ActivityAgent + EntityGeneratingActivity)
  #7  rtm:OutcomeSemanticsShape
  #8a rtm:ForwardTraceabilityShape
  #8b rtm:BackwardTraceabilityShape
  #10 Re-verification closure (runtime)

Shape #9 (NamedGraphIntegrityShape) is documented in the suite but
enforced by the audit module's cross-graph queries rather than pyshacl
(SHACL on rdflib Dataset operates per-graph).
"""

from __future__ import annotations

import warnings

import pytest
from rdflib import Dataset, Literal, URIRef
from rdflib.namespace import RDF, XSD

from ontology.prefixes import ADCS, EARL, G_ATTESTATIONS, G_EVIDENCE, GSN, PROV, RTM
from pipeline.runner import run_pipeline
from traceability.verification import (
    ReverificationMismatch,
    ShapeViolation,
    verify,
)


@pytest.fixture(scope="module")
def nominal_dataset() -> Dataset:
    """A fully attested pipeline run — should satisfy every shape."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return run_pipeline(auto_attest=True)


def _has_shape_violation(violations: list[ShapeViolation], shape_substring: str) -> bool:
    """True if any violation's source shape or message contains the substring."""
    return any(
        shape_substring in v.shape or shape_substring in v.message
        for v in violations
    )


# ---------------------------------------------------------------------------
# POSITIVE: the nominal pipeline graph satisfies the full suite
# ---------------------------------------------------------------------------

def test_nominal_pipeline_passes_all_shapes(nominal_dataset):
    report = verify(nominal_dataset, skip_reverification=False)
    assert report.conforms, (
        f"Nominal pipeline fails closure rules:\n{chr(10).join(report.summary_lines())}"
    )
    assert not report.shape_violations
    assert not report.reverification_mismatches


# ---------------------------------------------------------------------------
# Negative cases — each breaks one invariant and confirms the shape fires.
# Tests use a copy of the nominal Dataset so they don't pollute the fixture.
# ---------------------------------------------------------------------------

def _copy(ds: Dataset) -> Dataset:
    """Shallow Dataset copy for negative-case tests."""
    new = Dataset(default_union=True)
    for s, p, o, c in ds.quads():
        new.add((s, p, o, c))
    return new


# -- #1: AttestationShape ---------------------------------------------------

def test_attestation_missing_justification_fails(nominal_dataset):
    """Remove the gsn:Justification node from REQ-003's attestation; the
    AttestationShape's qualifiedMinCount=1 on gsn:inContextOf with class
    gsn:Justification must fire."""
    ds = _copy(nominal_dataset)
    att_g = ds.graph(URIRef(G_ATTESTATIONS))
    sufficiency = ADCS["sufficiency/ATT-REQ-003"]
    for s, p, o in list(att_g.triples((sufficiency, None, None))):
        att_g.remove((s, p, o))
    # Also remove the inContextOf link to that node
    att_g.remove((ADCS["ATT-REQ-003"], GSN.inContextOf, sufficiency))

    report = verify(ds, skip_reverification=True)
    assert not report.conforms
    assert _has_shape_violation(report.shape_violations, "sufficiency")


# -- #3: EvidenceShape ------------------------------------------------------

def test_evidence_missing_content_hash_fails(nominal_dataset):
    """An evidence artifact without rtm:contentHash violates EvidenceShape."""
    ds = _copy(nominal_dataset)
    ev_g = ds.graph(URIRef(G_EVIDENCE))
    ev = ADCS["EV-PROOF-REQ-003"]
    for o in list(ev_g.objects(ev, RTM.contentHash)):
        ev_g.remove((ev, RTM.contentHash, o))

    report = verify(ds, skip_reverification=True)
    assert not report.conforms
    assert _has_shape_violation(report.shape_violations, "contentHash")


# -- #5: GsnArgumentShape ---------------------------------------------------

def test_gsn_assumption_empty_statement_fails(nominal_dataset):
    """A gsn:Assumption with an empty gsn:statement violates GsnTextShape."""
    ds = _copy(nominal_dataset)
    att_g = ds.graph(URIRef(G_ATTESTATIONS))
    adequacy = ADCS["adequacy/ATT-REQ-003"]
    for o in list(att_g.objects(adequacy, GSN.statement)):
        att_g.remove((adequacy, GSN.statement, o))
    att_g.add((adequacy, GSN.statement, Literal("")))

    report = verify(ds, skip_reverification=True)
    assert not report.conforms
    assert _has_shape_violation(report.shape_violations, "gsn:statement")


# -- #6: ProvenanceShape ----------------------------------------------------

def test_activity_without_agent_fails(nominal_dataset):
    """A prov:Activity without prov:wasAssociatedWith violates ActivityAgentShape."""
    ds = _copy(nominal_dataset)
    # Add a fresh activity with no agent
    rogue = ADCS["exec/rogue-no-agent"]
    plan_exec = ds.graph(URIRef("http://example.org/adcs-demo/graph/plan-execution"))
    plan_exec.add((rogue, RDF.type, PROV.Activity))

    report = verify(ds, skip_reverification=True)
    assert not report.conforms
    assert _has_shape_violation(report.shape_violations, "wasAssociatedWith")


# -- #8a: ForwardTraceabilityShape -----------------------------------------

def test_forward_traceability_fails_when_attestation_removed(nominal_dataset):
    """Remove all attestation triples for REQ-002; forward shape must fire."""
    ds = _copy(nominal_dataset)
    att_g = ds.graph(URIRef(G_ATTESTATIONS))
    att = ADCS["ATT-REQ-002"]
    for s, p, o in list(att_g.triples((att, None, None))):
        att_g.remove((s, p, o))

    report = verify(ds, skip_reverification=True)
    assert not report.conforms
    assert _has_shape_violation(report.shape_violations, "Forward-traceability")


# -- #8b: BackwardTraceabilityShape ----------------------------------------

def test_backward_traceability_fails_when_addressing_removed(nominal_dataset):
    """An attestation references evidence that does NOT declare addressing
    the same requirement — backward shape must fire."""
    ds = _copy(nominal_dataset)
    ev_g = ds.graph(URIRef(G_EVIDENCE))
    # Remove the rtm:addresses link from EV-PROOF-REQ-002 to REQ-002.
    ev = ADCS["EV-PROOF-REQ-002"]
    for o in list(ev_g.objects(ev, RTM.addresses)):
        ev_g.remove((ev, RTM.addresses, o))

    report = verify(ds, skip_reverification=True)
    assert not report.conforms
    assert _has_shape_violation(report.shape_violations, "Backward-traceability")


# -- #2: PlanInstantiationShape --------------------------------------------

def test_plan_activity_without_corresponds_step_fails(nominal_dataset):
    """A p-plan:Activity without correspondsToStep violates PlanInstantiationShape."""
    ds = _copy(nominal_dataset)
    rogue = ADCS["exec/rogue-no-step"]
    plan_exec = ds.graph(URIRef("http://example.org/adcs-demo/graph/plan-execution"))
    from ontology.prefixes import P_PLAN
    plan_exec.add((rogue, RDF.type, P_PLAN.Activity))
    plan_exec.add((rogue, PROV.startedAtTime,
                   Literal("2026-05-14T00:00:00Z", datatype=XSD.dateTime)))
    plan_exec.add((rogue, PROV.wasAssociatedWith, ADCS["agent/pipeline-runner"]))

    report = verify(ds, skip_reverification=True)
    assert not report.conforms
    assert _has_shape_violation(report.shape_violations, "correspondsToStep")


# -- #10: Re-verification closure ------------------------------------------

def test_reverification_returns_no_mismatches_on_nominal(nominal_dataset):
    """Re-running every proof should produce hashes identical to those stored."""
    from traceability.verification import verify_reverification
    mismatches = verify_reverification(nominal_dataset)
    assert mismatches == [], (
        f"Re-verification mismatches on nominal: {mismatches}"
    )


# ---------------------------------------------------------------------------
# Stage 6.5 emits its activity (process-level confirmation that the
# validation step ran).
# ---------------------------------------------------------------------------

def test_validate_shapes_stage_emits_activity(nominal_dataset):
    """The pipeline records a ValidateShapes activity in <plan-execution>."""
    from ontology.prefixes import G_PLAN_EXECUTION, P_PLAN
    expected_step = URIRef(f"{RTM}plan/step/ValidateShapes")
    activities = [
        s for s, _, o, _ in nominal_dataset.quads(
            (None, P_PLAN.correspondsToStep, expected_step,
             URIRef(G_PLAN_EXECUTION))
        )
    ]
    assert len(activities) == 1, (
        f"Expected one ValidateShapes activity, got {len(activities)}"
    )
