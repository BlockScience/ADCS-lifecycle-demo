"""Tests for Phase 5-6: pipeline, interrogation, and reproducibility."""

from analysis.proof_scripts import ProofStatus
from interrogate.explain import explain_all, explain_requirement
from interrogate.reproduce import reproduce_all_evidence, reproduce_proof
from interrogate.visualize import build_dot
from pipeline.runner import run_pipeline
from pipeline.stages import LifecycleStage, check_gate


class TestStages:
    def test_stage_ordering(self):
        assert LifecycleStage.STRUCTURAL_DEFINED < LifecycleStage.ATTESTATION

    def test_gate_passes(self):
        check_gate(LifecycleStage.REPORTED, LifecycleStage.STRUCTURAL_DEFINED)

    def test_gate_fails(self):
        import pytest
        with pytest.raises(RuntimeError, match="gate violation"):
            check_gate(LifecycleStage.STRUCTURAL_DEFINED, LifecycleStage.REPORTED)


class TestPipelineState:
    def test_activity_to_stage_map_covers_known_steps(self):
        """Every step name emitted by the pipeline maps to a stage."""
        from pipeline.state import PipelineState
        from traceability.plan_execution import STEP_NAMES
        from rdflib import Dataset

        state = PipelineState(
            ds=Dataset(),
            compute_backend=None,  # type: ignore[arg-type]
            engineer_name="test",
        )
        # Stage 0 (OntologyAssembly) is emitted by stage0_assembly; all
        # other STEP_NAMES are emitted by pipeline.runner stages 1-8.
        missing = STEP_NAMES - set(state.activity_to_stage)
        assert not missing, f"Step names without stage map: {missing}"


class TestPipelineEndToEnd:
    def test_auto_pipeline(self):
        """Full pipeline with auto-attestation."""
        rtm = run_pipeline(auto_attest=True, engineer_name="Test Engineer")
        assert rtm is not None
        # Should have triples
        assert len(rtm) > 300

    def test_pipeline_no_attest(self):
        """Pipeline without attestation stage."""
        rtm = run_pipeline(skip_attestation=True)
        assert rtm is not None


class TestExplain:
    def test_explain_attested_requirement(self):
        rtm = run_pipeline(auto_attest=True, engineer_name="Dr. Test")
        explanation = explain_requirement(rtm, "REQ-003")
        assert "REQ-003" in explanation
        assert "Derived from" in explanation
        assert "Allocated to" in explanation
        assert "Evidence" in explanation
        assert "Attested by" in explanation

    def test_explain_declined_requirement(self):
        """REQ-001 should be attested with an earl:failed outcome
        (settling time not met). The declination is recorded as a
        well-formed attestation so the closure-rule suite can validate
        against an audit-complete graph."""
        rtm = run_pipeline(auto_attest=True, engineer_name="Dr. Test")
        explanation = explain_requirement(rtm, "REQ-001")
        assert "REQ-001" in explanation
        assert "earl:failed" in explanation, (
            "REQ-001 declination should surface as earl:failed in the explanation"
        )

    def test_explain_all(self):
        rtm = run_pipeline(auto_attest=True, engineer_name="Dr. Test")
        full = explain_all(rtm)
        assert "REQ-001" in full
        assert "REQ-002" in full
        assert "REQ-003" in full
        assert "REQ-004" in full

    def test_explain_shows_live_reverification(self):
        rtm = run_pipeline(auto_attest=True, engineer_name="Dr. Test")
        explanation = explain_requirement(rtm, "REQ-003")
        assert "Re-verification" in explanation


class TestReproduce:
    def test_reproduce_all_proofs(self):
        rtm = run_pipeline(auto_attest=True, engineer_name="Dr. Test")
        results = reproduce_all_evidence(rtm)

        assert len(results["proofs"]) == 4
        for proof in results["proofs"]:
            assert proof["status"] == ProofStatus.VERIFIED
            assert proof["hash_match"] is True

    def test_reproduce_all_simulations(self):
        rtm = run_pipeline(auto_attest=True, engineer_name="Dr. Test")
        results = reproduce_all_evidence(rtm)

        assert len(results["simulations"]) >= 2
        for sim in results["simulations"]:
            assert sim["reproduced"] is True


class TestVisualize:
    def test_build_rtm_figure(self):
        from interrogate.visualize import build_rtm_figure
        rtm = run_pipeline(auto_attest=True, engineer_name="Dr. Test")
        fig = build_rtm_figure(rtm)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_build_dot(self):
        rtm = run_pipeline(auto_attest=True, engineer_name="Dr. Test")
        dot = build_dot(rtm)
        assert "digraph RTM" in dot
        assert "REQ-001" in dot
        assert "REQ-003" in dot
        assert "satisfiedBy" in dot
        assert "derivedFrom" in dot
