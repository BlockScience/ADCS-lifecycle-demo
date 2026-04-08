"""Numerical ADCS simulation using scipy.

Integrates the full nonlinear quaternion + Euler dynamics with PD control
and gravity gradient disturbance. All parameters flow from the structural
RDF model via load_params.

The simulation produces observations *of the model*, not of the physical
system. Whether these observations constitute sufficient evidence for
requirement satisfaction is a matter for human attestation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class SimConfig:
    """Simulation configuration — everything needed to reproduce."""

    # Inertia (derived from structural model)
    Jxx: float
    Jyy: float
    Jzz: float
    # Controller gains
    Kp: float
    Kd: float
    # Orbital rate (for gravity gradient)
    n_orbital: float
    # Actuator limits
    max_torque: float
    max_momentum: float
    # Integration settings
    t_span: tuple[float, float]
    max_step: float = 0.5
    # Initial conditions
    q0: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    omega0: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        """Serialize for hashing."""
        return {
            "Jxx": self.Jxx, "Jyy": self.Jyy, "Jzz": self.Jzz,
            "Kp": self.Kp, "Kd": self.Kd,
            "n_orbital": self.n_orbital,
            "max_torque": self.max_torque, "max_momentum": self.max_momentum,
            "t_span": list(self.t_span), "max_step": self.max_step,
            "q0": list(self.q0), "omega0": list(self.omega0),
        }


@dataclass
class SimResult:
    """Simulation output — time histories of all state variables."""

    t: np.ndarray          # (N,) time vector
    q: np.ndarray          # (N, 4) quaternion [qx, qy, qz, qw]
    omega: np.ndarray      # (N, 3) angular velocity [rad/s]
    tau_ctrl: np.ndarray   # (N, 3) control torque [N.m]
    tau_gg: np.ndarray     # (N, 3) gravity gradient torque [N.m]
    h_wheel: np.ndarray    # (N, 3) wheel angular momentum [N.m.s]
    config: SimConfig

    def summary(self) -> dict:
        """Key metrics for hashing and reporting."""
        # Attitude error magnitude (small angle: 2*asin(|q_vec|) ≈ 2*|q_vec|)
        q_vec_norm = np.linalg.norm(self.q[:, :3], axis=1)
        theta_err_rad = 2 * q_vec_norm
        theta_err_deg = np.degrees(theta_err_rad)

        # Settling: find last time error exceeds 1% of initial
        if theta_err_deg[0] > 0:
            threshold = 0.01 * theta_err_deg[0]
            exceeded = np.where(theta_err_deg > threshold)[0]
            settling_idx = exceeded[-1] if len(exceeded) > 0 else 0
            settling_time = float(self.t[settling_idx])
        else:
            settling_time = 0.0

        # Peak wheel momentum
        h_mag = np.linalg.norm(self.h_wheel, axis=1)

        return {
            "final_error_deg": float(theta_err_deg[-1]),
            "peak_error_deg": float(np.max(theta_err_deg)),
            "settling_time_s": settling_time,
            "peak_wheel_momentum": float(np.max(h_mag)),
            "peak_control_torque": float(np.max(np.abs(self.tau_ctrl))),
            "final_omega_norm": float(np.linalg.norm(self.omega[-1])),
            "duration_s": float(self.t[-1]),
            "n_steps": len(self.t),
        }


def _quaternion_error(q: np.ndarray) -> np.ndarray:
    """Attitude error as vector part of quaternion (small angle approx)."""
    return q[:3]  # qx, qy, qz — for identity target, this IS the error


def _gravity_gradient_torque(
    q: np.ndarray, J: np.ndarray, n: float,
) -> np.ndarray:
    """Linearized gravity gradient torque at GEO.

    tau_gg_x = 3*n^2*(Jzz - Jyy)*theta_x
    tau_gg_y = 3*n^2*(Jzz - Jxx)*theta_y  (note sign)
    tau_gg_z = 0
    """
    theta = _quaternion_error(q)  # small angle
    return np.array([
        3 * n**2 * (J[2] - J[1]) * theta[0],
        3 * n**2 * (J[2] - J[0]) * theta[1],
        0.0,
    ])


def _pd_control(
    q: np.ndarray, omega: np.ndarray, Kp: float, Kd: float, max_torque: float,
) -> np.ndarray:
    """PD attitude control law with torque saturation.

    tau = -Kp/2 * q_err - Kd * omega, clamped to max_torque.
    """
    theta_err = _quaternion_error(q)
    tau = -Kp / 2 * theta_err - Kd * omega
    return np.clip(tau, -max_torque, max_torque)


def _quaternion_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Quaternion kinematics: dq/dt = 0.5 * Omega(omega) * q."""
    qx, qy, qz, qw = q
    wx, wy, wz = omega
    return 0.5 * np.array([
        qw * wx - qz * wy + qy * wz,
        qz * wx + qw * wy - qx * wz,
        -qy * wx + qx * wy + qw * wz,
        -qx * wx - qy * wy - qz * wz,
    ])


def _euler_equations(
    omega: np.ndarray, J: np.ndarray, tau: np.ndarray,
) -> np.ndarray:
    """Euler's rotation equations: J * domega/dt = tau - omega x (J*omega)."""
    Jw = J * omega  # element-wise for diagonal J
    cross = np.cross(omega, Jw)
    return (tau - cross) / J


def simulate_adcs(config: SimConfig) -> SimResult:
    """Integrate ADCS dynamics with PD control and gravity gradient."""
    J = np.array([config.Jxx, config.Jyy, config.Jzz])

    # Storage for torques and momentum (populated via dense output)
    tau_ctrl_list = []
    tau_gg_list = []
    h_wheel_list = []

    def dynamics(t, state):
        q = state[:4]
        omega = state[4:7]

        # Normalize quaternion to prevent drift
        q = q / np.linalg.norm(q)

        # Torques
        tau_c = _pd_control(q, omega, config.Kp, config.Kd, config.max_torque)
        tau_g = _gravity_gradient_torque(q, J, config.n_orbital)
        tau_total = tau_c + tau_g

        # Derivatives
        dq = _quaternion_derivative(q, omega)
        domega = _euler_equations(omega, J, tau_total)

        return np.concatenate([dq, domega])

    # Initial state
    x0 = np.array([*config.q0, *config.omega0])

    sol = solve_ivp(
        dynamics,
        config.t_span,
        x0,
        method="RK45",
        max_step=config.max_step,
        rtol=1e-10,
        atol=1e-12,
        dense_output=True,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # Extract results
    t = sol.t
    q = sol.y[:4].T  # (N, 4)
    omega = sol.y[4:7].T  # (N, 3)

    # Normalize quaternions
    q_norms = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / q_norms

    # Recompute torques at each output time
    tau_ctrl = np.zeros_like(omega)
    tau_gg = np.zeros_like(omega)
    h_wheel = np.zeros_like(omega)

    h_accumulated = np.zeros(3)
    dt_arr = np.diff(t, prepend=t[0])

    for i in range(len(t)):
        tau_ctrl[i] = _pd_control(
            q[i], omega[i], config.Kp, config.Kd, config.max_torque,
        )
        tau_gg[i] = _gravity_gradient_torque(q[i], J, config.n_orbital)
        # Wheel momentum = integral of control torque
        if i > 0:
            h_accumulated += tau_ctrl[i] * dt_arr[i]
        h_wheel[i] = h_accumulated

    return SimResult(
        t=t, q=q, omega=omega,
        tau_ctrl=tau_ctrl, tau_gg=tau_gg, h_wheel=h_wheel,
        config=config,
    )


def make_config_from_params(
    params: dict[str, float],
    t_end: float = 300.0,
    initial_error_deg: float = 10.0,
) -> SimConfig:
    """Build SimConfig from structural parameters.

    Default scenario: 10-degree initial attitude error, 300s simulation.
    """
    from analysis.symbolic import evaluate_inertia
    Jxx, Jyy, Jzz = evaluate_inertia(params)

    # Initial quaternion for small-angle rotation about X axis
    theta_rad = np.radians(initial_error_deg)
    q0 = (np.sin(theta_rad / 2), 0.0, 0.0, np.cos(theta_rad / 2))

    return SimConfig(
        Jxx=Jxx, Jyy=Jyy, Jzz=Jzz,
        Kp=params["Kp"], Kd=params["Kd"],
        n_orbital=params["orbitalRate"],
        max_torque=params["maxTorque"],
        max_momentum=params["maxMomentum"],
        t_span=(0.0, t_end),
        q0=q0,
    )


def run_step_response(params: dict[str, float]) -> SimResult:
    """Standard step response: 10-degree slew, 300s simulation."""
    config = make_config_from_params(params, t_end=300.0, initial_error_deg=10.0)
    return simulate_adcs(config)


def run_disturbance_rejection(params: dict[str, float]) -> SimResult:
    """Disturbance rejection: start at zero error, observe GG effects."""
    config = make_config_from_params(params, t_end=600.0, initial_error_deg=0.001)
    return simulate_adcs(config)
