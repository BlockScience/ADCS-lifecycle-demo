"""Tests for Phase 4: evidence binding, RTM assembly, and attestation."""

from rdflib import Graph

from analysis.build_proofs import build_all_proofs
from analysis.load_params import load_params, load_structural_graph
from analysis.numerical import run_step_response
from analysis.proof_scripts import ProofStatus, verify_proof
from evidence.binding import (
    bind_computation_engines,
    bind_proof_evidence,
    bind_simulation_evidence,
)
from evidence.hashing import (
    hash_evidence,
    hash_proof,
    hash_simulation,
    hash_structural_model,
)
from ontology.prefixes import bind_prefixes
from traceability.attestation import request_attestation
from traceability.queries import (
    ALL_ATTESTATIONS,
    ALL_EVIDENCE,
    BACKWARD_TRACE,
    FORWARD_TRACE,
    query_to_dicts,
)
from traceability.rtm import (
    assemble_rtm,
    get_unattested_requirements,
    load_base_graph,
    print_rtm_summary,
    verify_evidence_completeness,
    verify_structural_completeness,
)


def _build_full_rtm_with_evidence() -> Graph:
    """Helper: build a complete RTM with evidence for all 4 requirements."""
    base = load_base_graph()
    struct_graph = load_structural_graph()
    model_hash = hash_structural_model(struct_graph)
    params = load_params(struct_graph)

    # Build proof evidence
    proofs = build_all_proofs(model_hash)
    ev_graph = Graph()
    bind_prefixes(ev_graph)
    bind_computation_engines(ev_graph)

    for req_id, script in proofs.items():
        result = verify_proof(script, model_hash)
        assert result.status == ProofStatus.VERIFIED
        p_hash = hash_proof(script, model_hash)
        c_hash = hash_evidence(model_hash, proof_hash=p_hash)

        bind_proof_evidence(
            ev_graph,
            evidence_id=f"EV-PROOF-{req_id}",
            activity_id=f"SA-{req_id}",
            requirement_id=req_id,
            model_hash=model_hash,
            proof_hash=p_hash,
            content_hash=c_hash,
            result_summary=f"Symbolic proof for {req_id}: {script.claim}",
            source_file="analysis/build_proofs.py",
        )

    # Build simulation evidence for REQ-001 and REQ-002
    sim_result = run_step_response(params)
    summary = sim_result.summary()
    sim_h = hash_simulation(sim_result.config.to_dict(), summary)

    for req_id, desc in [
        ("REQ-001", f"Step response: settling={summary['settling_time_s']:.1f}s, "
                    f"final_error={summary['final_error_deg']:.4f} deg"),
        ("REQ-002", f"Peak wheel momentum: {summary['peak_wheel_momentum']:.3f} N.m.s"),
    ]:
        bind_simulation_evidence(
            ev_graph,
            evidence_id=f"EV-SIM-{req_id}",
            activity_id=f"NS-{req_id}",
            requirement_id=req_id,
            model_hash=model_hash,
            sim_hash=sim_h,
            result_summary=desc,
            source_file="analysis/numerical.py",
        )

    return assemble_rtm(base, ev_graph)


class TestStructuralCompleteness:
    def test_all_requirements_have_satisfy_links(self):
        graph = load_base_graph()
        issues = verify_structural_completeness(graph)
        assert issues == [], f"Structural issues: {issues}"


class TestEvidenceBinding:
    def test_evidence_nodes_created(self):
        rtm = _build_full_rtm_with_evidence()
        evidence = query_to_dicts(rtm, ALL_EVIDENCE)
        # 4 proof artifacts + 2 simulation results = 6
        assert len(evidence) >= 6, f"Expected >= 6 evidence nodes, got {len(evidence)}"

    def test_evidence_has_hashes(self):
        rtm = _build_full_rtm_with_evidence()
        evidence = query_to_dicts(rtm, ALL_EVIDENCE)
        for ev in evidence:
            assert ev["hash"] is not None and len(ev["hash"]) > 10

    def test_evidence_completeness(self):
        rtm = _build_full_rtm_with_evidence()
        issues = verify_evidence_completeness(rtm)
        assert issues == [], f"Evidence issues: {issues}"


class TestAttestation:
    def test_auto_attestation(self):
        rtm = _build_full_rtm_with_evidence()
        att_uri = request_attestation(
            rtm, "REQ-003", "Dr. Test Engineer",
            auto_attest=True,
            model_adequacy="Linearized model adequate for stability analysis.",
            evidence_sufficiency="Routh-Hurwitz proof confirms stability.",
        )
        assert att_uri is not None

    def test_attestation_recorded_in_graph(self):
        rtm = _build_full_rtm_with_evidence()
        request_attestation(
            rtm, "REQ-003", "Dr. Test Engineer",
            auto_attest=True,
            model_adequacy="Model adequate.",
            evidence_sufficiency="Evidence sufficient.",
        )
        attestations = query_to_dicts(rtm, ALL_ATTESTATIONS)
        assert len(attestations) == 1
        assert attestations[0]["reqName"] == "REQ-003"
        assert attestations[0]["engineer"] == "Dr. Test Engineer"

    def test_unattested_requirements(self):
        rtm = _build_full_rtm_with_evidence()
        # Before any attestation
        unattested = get_unattested_requirements(rtm)
        assert set(unattested) == {"REQ-001", "REQ-002", "REQ-003", "REQ-004"}

        # After attesting REQ-003
        request_attestation(
            rtm, "REQ-003", "Dr. Test",
            auto_attest=True,
            model_adequacy="ok", evidence_sufficiency="ok",
        )
        unattested = get_unattested_requirements(rtm)
        assert "REQ-003" not in unattested
        assert len(unattested) == 3


class TestForwardBackwardTrace:
    def test_forward_trace(self):
        rtm = _build_full_rtm_with_evidence()
        rows = query_to_dicts(rtm, FORWARD_TRACE % "REQ-003")
        assert len(rows) > 0
        # Should have evidence linked
        ev_hashes = [r["evHash"] for r in rows if r["evHash"]]
        assert len(ev_hashes) > 0, "Forward trace found no evidence for REQ-003"

    def test_backward_trace_after_attestation(self):
        rtm = _build_full_rtm_with_evidence()
        request_attestation(
            rtm, "REQ-003", "Dr. Test",
            auto_attest=True,
            model_adequacy="ok", evidence_sufficiency="ok",
        )
        rows = query_to_dicts(rtm, BACKWARD_TRACE)
        req_names = [r["reqName"] for r in rows]
        assert "REQ-003" in req_names

    def test_attestation_links_evidence(self):
        """Attestation should link to the evidence it reviewed."""
        rtm = _build_full_rtm_with_evidence()
        request_attestation(
            rtm, "REQ-001", "Dr. Test",
            auto_attest=True,
            model_adequacy="ok", evidence_sufficiency="ok",
        )
        rows = query_to_dicts(rtm, BACKWARD_TRACE)
        req001_rows = [r for r in rows if r["reqName"] == "REQ-001"]
        assert len(req001_rows) > 0, "Backward trace found no evidence for REQ-001 attestation"


class TestRTMSummary:
    def test_print_summary(self):
        """A graph with evidence but no attestations should surface as
        "NO ATTESTATION" — the trace is incomplete, distinct from a
        requirement that's attested with earl:failed."""
        rtm = _build_full_rtm_with_evidence()
        summary = print_rtm_summary(rtm)
        assert "REQ-001" in summary
        assert "NO ATTESTATION" in summary, (
            "Pre-attestation requirements should be reported as 'NO ATTESTATION', "
            "not 'UNATTESTED' — the new framing distinguishes trace gaps from "
            "engineering findings (earl:failed)."
        )

    def test_summary_reflects_attestation(self):
        rtm = _build_full_rtm_with_evidence()
        request_attestation(
            rtm, "REQ-003", "Dr. Test",
            auto_attest=True,
            model_adequacy="ok", evidence_sufficiency="ok",
        )
        summary = print_rtm_summary(rtm)
        assert "ATTESTED" in summary
