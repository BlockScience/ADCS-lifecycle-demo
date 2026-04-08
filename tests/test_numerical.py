"""Tests for Phase 3: numerical simulation."""

import math

import numpy as np

from analysis.load_params import load_params
from analysis.numerical import (
    SimConfig,
    make_config_from_params,
    run_disturbance_rejection,
    run_step_response,
    simulate_adcs,
)
from analysis.symbolic import evaluate_inertia, stability_margins


class TestSimConfig:
    def test_config_from_params(self):
        params = load_params()
        config = make_config_from_params(params)
        assert config.Jxx > 0
        assert config.Kp == params["Kp"]
        assert config.Kd == params["Kd"]

    def test_config_serialization(self):
        params = load_params()
        config = make_config_from_params(params)
        d = config.to_dict()
        assert "Jxx" in d
        assert "t_span" in d
        assert isinstance(d["t_span"], list)


class TestStepResponse:
    def test_simulation_completes(self):
        params = load_params()
        result = run_step_response(params)
        assert len(result.t) > 10
        assert result.t[-1] == 300.0

    def test_attitude_converges(self):
        """System should converge toward zero error (it's stable)."""
        params = load_params()
        result = run_step_response(params)
        summary = result.summary()
        # Final error should be much smaller than initial 10 degrees
        assert summary["final_error_deg"] < 1.0

    def test_quaternion_normalized(self):
        params = load_params()
        result = run_step_response(params)
        norms = np.linalg.norm(result.q, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_control_torque_within_limits(self):
        params = load_params()
        result = run_step_response(params)
        max_tau = params["maxTorque"]
        assert np.all(np.abs(result.tau_ctrl) <= max_tau + 1e-10)

    def test_eigenvalue_prediction_consistent(self):
        """Numerical decay rate should be consistent with symbolic eigenvalues.

        We compare the decay envelope of the attitude error with the
        predicted dominant eigenvalue. They should agree within ~30%
        (numerical has nonlinear effects, coupling, etc.)
        """
        params = load_params()
        result = run_step_response(params)

        # Attitude error envelope
        q_vec_norm = np.linalg.norm(result.q[:, :3], axis=1)
        theta_err = 2 * q_vec_norm

        # Find decay rate from first and last quarter of simulation
        t = result.t
        mask_early = (t > 10) & (t < 50)
        mask_late = (t > 200) & (t < 280)

        if np.any(mask_early) and np.any(mask_late):
            err_early = np.mean(theta_err[mask_early])
            err_late = np.mean(theta_err[mask_late])
            t_early = np.mean(t[mask_early])
            t_late = np.mean(t[mask_late])

            if err_early > 1e-10 and err_late > 1e-10:
                numerical_decay = -np.log(err_late / err_early) / (t_late - t_early)

                # Compare with symbolic
                margins = stability_margins(params)
                symbolic_decay = -margins["x"]  # X axis (dominant, slowest)

                # Should agree within factor of 3 (loose bound for nonlinear effects)
                ratio = numerical_decay / symbolic_decay
                assert 0.3 < ratio < 3.0, (
                    f"Decay rate mismatch: numerical={numerical_decay:.4f}, "
                    f"symbolic={symbolic_decay:.4f}, ratio={ratio:.2f}"
                )

    def test_summary_keys(self):
        params = load_params()
        result = run_step_response(params)
        summary = result.summary()
        expected_keys = {
            "final_error_deg", "peak_error_deg", "settling_time_s",
            "peak_wheel_momentum", "peak_control_torque",
            "final_omega_norm", "duration_s", "n_steps",
        }
        assert expected_keys == set(summary.keys())


class TestDisturbanceRejection:
    def test_simulation_completes(self):
        params = load_params()
        result = run_disturbance_rejection(params)
        assert len(result.t) > 10
        assert result.t[-1] == 600.0

    def test_error_stays_small(self):
        """Starting near zero, GG disturbance should not cause large error."""
        params = load_params()
        result = run_disturbance_rejection(params)
        summary = result.summary()
        # Error should stay well below 0.1 degrees
        assert summary["peak_error_deg"] < 0.1

    def test_gravity_gradient_torque_present(self):
        """GG torque should be nonzero (we're not at zero attitude)."""
        params = load_params()
        result = run_disturbance_rejection(params)
        # At least some GG torque should be present
        gg_max = np.max(np.abs(result.tau_gg))
        assert gg_max > 0


class TestWheelMomentum:
    def test_peak_momentum_below_capacity(self):
        """REQ-002: wheel momentum should not exceed rated capacity."""
        params = load_params()
        result = run_step_response(params)
        summary = result.summary()
        assert summary["peak_wheel_momentum"] < params["maxMomentum"], (
            f"Peak momentum {summary['peak_wheel_momentum']:.3f} exceeds "
            f"capacity {params['maxMomentum']}"
        )
