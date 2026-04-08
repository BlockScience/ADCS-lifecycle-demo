"""Tests for Phase 2: symbolic analysis, proof scripts, and hashing."""

import math

from analysis.build_proofs import build_all_proofs
from analysis.load_params import load_params, load_structural_graph
from analysis.proof_scripts import ProofStatus, verify_proof
from analysis.symbolic import (
    build_inertia_tensor_symbolic,
    evaluate_eigenvalues,
    evaluate_gravity_gradient,
    evaluate_inertia,
    evaluate_pointing_budget,
    evaluate_wheel_momentum,
    run_symbolic_analysis,
    stability_margins,
)
from evidence.hashing import hash_proof, hash_structural_model


class TestLoadParams:
    def test_load_params_returns_all_expected_keys(self):
        params = load_params()
        expected_keys = {
            "mass", "bodyLength", "bodyWidth", "bodyHeight",
            "panelSpan", "panelChord", "panelMass",
            "antennaDiameter", "antennaOffset", "antennaMass",
            "maxTorque", "maxMomentum", "wheelMass",
            "stAccuracy", "stUpdateRate", "stMass",
            "gyroARW", "gyroBias", "controlBandwidth", "imuMass",
            "Kp", "Kd", "orbitalRate",
        }
        assert expected_keys.issubset(set(params.keys()))

    def test_param_values_are_positive(self):
        params = load_params()
        for key, val in params.items():
            assert val > 0, f"{key} should be positive, got {val}"


class TestInertia:
    def test_symbolic_inertia_returns_three_components(self):
        Ixx, Iyy, Izz = build_inertia_tensor_symbolic()
        assert Ixx is not None
        assert Iyy is not None
        assert Izz is not None

    def test_numerical_inertia_reasonable(self):
        params = load_params()
        Ixx, Iyy, Izz = evaluate_inertia(params)
        # All should be positive and in a reasonable range for a satellite
        for val in [Ixx, Iyy, Izz]:
            assert 50 < val < 1000, f"Inertia {val} out of expected range"
        # Panels dominate Ixx and Izz, so they should be larger than Iyy
        assert Ixx > Iyy
        assert Izz > Iyy


class TestStability:
    def test_all_eigenvalues_have_negative_real_parts(self):
        params = load_params()
        eigs = evaluate_eigenvalues(params)
        for axis, eig_list in eigs.items():
            for eig in eig_list:
                assert eig.real < 0, f"Axis {axis}: eigenvalue {eig} not stable"

    def test_stability_margins_meet_req003(self):
        """REQ-003: Re(lambda) <= -0.010 rad/s"""
        params = load_params()
        margins = stability_margins(params)
        for axis, worst_re in margins.items():
            assert worst_re <= -0.010, (
                f"Axis {axis}: Re(lambda)={worst_re} violates REQ-003 threshold"
            )


class TestPointingBudget:
    def test_pointing_error_below_requirement(self):
        """REQ-001: pointing error < 0.1 degrees"""
        params = load_params()
        budget = evaluate_pointing_budget(params)
        assert budget["theta_ss_deg"] < 0.1

    def test_settling_time_computed(self):
        """Settling time is computed and finite (whether it meets REQ-001
        is a matter for human attestation, not automated testing)."""
        params = load_params()
        budget = evaluate_pointing_budget(params)
        assert budget["settling_time_s"] > 0
        assert budget["settling_time_s"] < float("inf")


class TestGravityGradient:
    def test_gg_torque_below_actuator_capacity(self):
        """REQ-004: disturbance torque < max torque"""
        params = load_params()
        gg = evaluate_gravity_gradient(params)
        assert gg["tau_gg_x"] < gg["tau_max"]
        assert gg["tau_gg_y"] < gg["tau_max"]

    def test_gg_torque_is_small(self):
        """GEO gravity gradient should be very small"""
        params = load_params()
        gg = evaluate_gravity_gradient(params)
        assert gg["tau_gg_x"] < 1e-3  # micro-Newton-meter range
        assert gg["tau_gg_y"] < 1e-3


class TestWheelMomentum:
    def test_peak_momentum_below_capacity(self):
        """REQ-002: peak momentum < maxMomentum"""
        params = load_params()
        wm = evaluate_wheel_momentum(params)
        assert wm["h_peak"] < wm["h_max"], (
            f"Peak momentum {wm['h_peak']:.3f} exceeds capacity {wm['h_max']}"
        )


class TestHashing:
    def test_model_hash_deterministic(self):
        g1 = load_structural_graph()
        g2 = load_structural_graph()
        assert hash_structural_model(g1) == hash_structural_model(g2)

    def test_model_hash_is_64_hex_chars(self):
        g = load_structural_graph()
        h = hash_structural_model(g)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestProofScripts:
    def test_all_proofs_verify(self):
        g = load_structural_graph()
        model_hash = hash_structural_model(g)
        proofs = build_all_proofs(model_hash)

        for req_id, script in proofs.items():
            result = verify_proof(script, model_hash)
            assert result.status == ProofStatus.VERIFIED, (
                f"{req_id} proof failed: {result.failure_summary}"
            )

    def test_proof_hash_deterministic(self):
        g = load_structural_graph()
        model_hash = hash_structural_model(g)
        proofs = build_all_proofs(model_hash)

        for req_id, script in proofs.items():
            h1 = hash_proof(script, model_hash)
            h2 = hash_proof(script, model_hash)
            assert h1 == h2, f"{req_id}: proof hash not deterministic"

    def test_proof_hash_changes_with_model_hash(self):
        g = load_structural_graph()
        model_hash = hash_structural_model(g)
        proofs = build_all_proofs(model_hash)

        fake_hash = "0" * 64
        fake_proofs = build_all_proofs(fake_hash)

        for req_id in proofs:
            h_real = hash_proof(proofs[req_id], model_hash)
            h_fake = hash_proof(fake_proofs[req_id], fake_hash)
            assert h_real != h_fake

    def test_proof_round_trip_serialization(self):
        g = load_structural_graph()
        model_hash = hash_structural_model(g)
        proofs = build_all_proofs(model_hash)

        from analysis.proof_scripts import ProofScript

        for req_id, script in proofs.items():
            evidence = script.to_evidence()
            restored = ProofScript.from_evidence(evidence)
            result = verify_proof(restored, model_hash)
            assert result.status == ProofStatus.VERIFIED, (
                f"{req_id} round-trip failed: {result.failure_summary}"
            )

    def test_wrong_model_hash_fails(self):
        g = load_structural_graph()
        model_hash = hash_structural_model(g)
        proofs = build_all_proofs(model_hash)

        wrong_hash = "f" * 64
        for req_id, script in proofs.items():
            result = verify_proof(script, wrong_hash)
            assert result.status == ProofStatus.FAILED
            assert "mismatch" in result.failure_summary


class TestCollectedAnalysis:
    def test_run_symbolic_analysis_succeeds(self):
        params = load_params()
        result = run_symbolic_analysis(params)
        assert len(result.inertia) == 3
        assert len(result.eigenvalues) == 3
        assert len(result.stability_margins) == 3
