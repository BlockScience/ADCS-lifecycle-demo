"""Symbolic ADCS analysis using SymPy.

All functions accept a parameter dict (from load_params) and return
SymPy expressions or evaluated results. Nothing is hardcoded — every
physical quantity flows from the structural RDF model.

Key analyses:
- Composite inertia tensor via parallel axis theorem
- Closed-loop eigenvalues for PD controller (per-axis linearization)
- Steady-state pointing error via final value theorem
- Gravity gradient disturbance torque at GEO
- Wheel momentum bounds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp


# ---------------------------------------------------------------------------
# Symbolic parameter symbols (used in proof scripts)
# ---------------------------------------------------------------------------

# Bus
m_bus, Lx, Ly, Lz = sp.symbols("m_bus L_x L_y L_z", positive=True)
# Panels
m_panel, span, chord = sp.symbols("m_panel span chord", positive=True)
# Antenna
m_ant, d_ant, z_ant = sp.symbols("m_ant d_ant z_ant", positive=True, real=True)
# Controller
Kp, Kd = sp.symbols("K_p K_d", positive=True)
# Orbital rate
n = sp.Symbol("n", positive=True)
# Inertia (derived)
Jxx, Jyy, Jzz = sp.symbols("J_xx J_yy J_zz", positive=True)
# Eigenvalue
s = sp.Symbol("s")
# Wheel
h_max = sp.Symbol("h_max", positive=True)
tau_max = sp.Symbol("tau_max", positive=True)


# ---------------------------------------------------------------------------
# Inertia tensor — parallel axis theorem
# ---------------------------------------------------------------------------

def _box_inertia(mass, lx, ly, lz):
    """Inertia of a uniform rectangular box about its center of mass."""
    Ixx = mass * (ly**2 + lz**2) / 12
    Iyy = mass * (lx**2 + lz**2) / 12
    Izz = mass * (lx**2 + ly**2) / 12
    return sp.Matrix([Ixx, Iyy, Izz])


def _thin_plate_inertia(mass, span_dim, chord_dim):
    """Inertia of a thin flat plate (panel) about its center of mass.

    span_dim along Y, chord_dim along X, negligible thickness along Z.
    """
    Ixx = mass * span_dim**2 / 12
    Iyy = mass * chord_dim**2 / 12
    Izz = mass * (span_dim**2 + chord_dim**2) / 12
    return sp.Matrix([Ixx, Iyy, Izz])


def _disk_inertia(mass, diameter):
    """Inertia of a thin disk about its center of mass.

    Disk in XY plane, Z is the symmetry axis.
    """
    r = diameter / 2
    Ixx = mass * r**2 / 4
    Iyy = mass * r**2 / 4
    Izz = mass * r**2 / 2
    return sp.Matrix([Ixx, Iyy, Izz])


def _parallel_axis(I_cm, mass, offset):
    """Parallel axis theorem: shift inertia by offset vector [dx, dy, dz].

    offset is a 3-vector; adds mass*(|d|^2*I - d*d^T) diagonal terms.
    For diagonal inertia tensors, the diagonal correction is:
        delta_Ixx = mass * (dy^2 + dz^2)
        delta_Iyy = mass * (dx^2 + dz^2)
        delta_Izz = mass * (dx^2 + dy^2)
    """
    dx, dy, dz = offset
    delta = sp.Matrix([
        mass * (dy**2 + dz**2),
        mass * (dx**2 + dz**2),
        mass * (dx**2 + dy**2),
    ])
    return I_cm + delta


def build_inertia_tensor_symbolic():
    """Build the composite inertia tensor symbolically.

    Returns (Ixx_expr, Iyy_expr, Izz_expr) as SymPy expressions
    in terms of the symbolic parameter symbols.
    """
    # Bus: centered at origin
    I_bus = _box_inertia(m_bus, Lx, Ly, Lz)

    # Two symmetric panels offset along Y
    # Panel center is at y = Ly/2 + span/2
    y_panel = Ly / 2 + span / 2
    I_panel_cm = _thin_plate_inertia(m_panel, span, chord)
    I_panel = _parallel_axis(I_panel_cm, m_panel, [0, y_panel, 0])
    # Two panels (symmetric about Y=0, same Ixx/Iyy/Izz contribution)
    I_panels = 2 * I_panel

    # Antenna: offset along Z
    I_ant_cm = _disk_inertia(m_ant, d_ant)
    I_ant = _parallel_axis(I_ant_cm, m_ant, [0, 0, z_ant])

    I_total = I_bus + I_panels + I_ant
    return sp.simplify(I_total[0]), sp.simplify(I_total[1]), sp.simplify(I_total[2])


def evaluate_inertia(params: dict[str, float]) -> tuple[float, float, float]:
    """Numerically evaluate the composite inertia tensor."""
    Ixx_sym, Iyy_sym, Izz_sym = build_inertia_tensor_symbolic()
    subs = {
        m_bus: params["mass"],
        Lx: params["bodyLength"],
        Ly: params["bodyWidth"],
        Lz: params["bodyHeight"],
        m_panel: params["panelMass"],
        span: params["panelSpan"],
        chord: params["panelChord"],
        m_ant: params["antennaMass"],
        d_ant: params["antennaDiameter"],
        z_ant: params["antennaOffset"],
    }
    return (
        float(Ixx_sym.subs(subs)),
        float(Iyy_sym.subs(subs)),
        float(Izz_sym.subs(subs)),
    )


# ---------------------------------------------------------------------------
# Closed-loop stability — per-axis linearized eigenvalue analysis
# ---------------------------------------------------------------------------

def characteristic_polynomial_single_axis(J_axis):
    """Characteristic polynomial for one axis of PD-controlled rigid body.

    Linearized about zero attitude error:
        J * d²θ/dt² + Kd * dθ/dt + Kp/2 * θ = 0

    (The 1/2 factor on Kp comes from quaternion error feedback.)

    Returns the polynomial in s.
    """
    return J_axis * s**2 + Kd * s + Kp / 2


def eigenvalues_single_axis(J_axis):
    """Closed-loop eigenvalues for a single axis."""
    poly = characteristic_polynomial_single_axis(J_axis)
    return sp.solve(poly, s)


def evaluate_eigenvalues(params: dict[str, float]) -> dict[str, list[complex]]:
    """Compute eigenvalues for all three axes numerically."""
    Ixx_val, Iyy_val, Izz_val = evaluate_inertia(params)
    kp_val = params["Kp"]
    kd_val = params["Kd"]

    result = {}
    for axis, J_val in [("x", Ixx_val), ("y", Iyy_val), ("z", Izz_val)]:
        eigs = eigenvalues_single_axis(Jxx)
        eigs_num = [complex(e.subs({Jxx: J_val, Kp: kp_val, Kd: kd_val})) for e in eigs]
        result[axis] = eigs_num
    return result


def stability_margins(params: dict[str, float]) -> dict[str, float]:
    """Compute Re(lambda) for each axis — must be <= -0.010 for REQ-003."""
    eigs = evaluate_eigenvalues(params)
    margins = {}
    for axis, eig_list in eigs.items():
        # The eigenvalue with largest (least negative) real part
        worst_re = max(e.real for e in eig_list)
        margins[axis] = worst_re
    return margins


# ---------------------------------------------------------------------------
# Gravity gradient disturbance torque
# ---------------------------------------------------------------------------

def gravity_gradient_torque_symbolic():
    """Gravity gradient torque at GEO (linearized, per-axis).

    tau_gg_x = 3 * n^2 * (Jzz - Jyy) * theta_x
    tau_gg_y = 3 * n^2 * (Jzz - Jxx) * theta_y  (note: can be negative)
    tau_gg_z = 0

    Returns max magnitude torques assuming worst-case theta = 0.1 deg.
    """
    theta_max = sp.Rational(1, 10) * sp.pi / 180  # 0.1 degrees in radians
    tau_x = 3 * n**2 * sp.Abs(Jzz - Jyy) * theta_max
    tau_y = 3 * n**2 * sp.Abs(Jzz - Jxx) * theta_max
    return tau_x, tau_y


def evaluate_gravity_gradient(params: dict[str, float]) -> dict[str, float]:
    """Numerically evaluate gravity gradient torque magnitudes."""
    Ixx_val, Iyy_val, Izz_val = evaluate_inertia(params)
    n_val = params["orbitalRate"]

    tau_x_sym, tau_y_sym = gravity_gradient_torque_symbolic()
    subs = {Jxx: Ixx_val, Jyy: Iyy_val, Jzz: Izz_val, n: n_val}
    return {
        "tau_gg_x": float(tau_x_sym.subs(subs)),
        "tau_gg_y": float(tau_y_sym.subs(subs)),
        "tau_max": params["maxTorque"],
    }


# ---------------------------------------------------------------------------
# Steady-state pointing error (final value theorem)
# ---------------------------------------------------------------------------

def steady_state_error_symbolic():
    """Steady-state pointing error for step disturbance.

    For a PD controller with gravity gradient disturbance,
    the steady-state error is: theta_ss = tau_gg / (Kp/2)

    This is because the PD controller has a finite DC gain of Kp/2.
    """
    tau_gg = sp.Symbol("tau_gg", positive=True)
    return 2 * tau_gg / Kp


def evaluate_pointing_budget(params: dict[str, float]) -> dict[str, float]:
    """Compute pointing error budget.

    Combines gravity gradient steady-state error with star tracker noise floor.
    """
    gg = evaluate_gravity_gradient(params)
    kp_val = params["Kp"]

    # Worst-case gravity gradient torque
    tau_gg_max = max(gg["tau_gg_x"], gg["tau_gg_y"])

    # Steady-state error from gravity gradient (radians)
    theta_ss_rad = 2 * tau_gg_max / kp_val
    theta_ss_deg = theta_ss_rad * 180 / 3.141592653589793

    # Star tracker noise floor
    st_accuracy_arcsec = params["stAccuracy"]
    st_floor_deg = st_accuracy_arcsec / 3600.0

    # Settling time estimate from dominant eigenvalue
    margins = stability_margins(params)
    worst_re = max(margins.values())  # least negative
    settling_time = -4.0 / worst_re if worst_re < 0 else float("inf")

    return {
        "theta_ss_rad": theta_ss_rad,
        "theta_ss_deg": theta_ss_deg,
        "st_floor_deg": st_floor_deg,
        "settling_time_s": settling_time,
    }


# ---------------------------------------------------------------------------
# Wheel momentum bounds
# ---------------------------------------------------------------------------

def wheel_momentum_bound_symbolic():
    """Symbolic bound on peak wheel momentum.

    For a step response from theta_0 to 0, the peak wheel momentum is
    approximately: h_peak = Kd * omega_peak, where omega_peak ~ theta_0 * |lambda|

    More conservatively, for the PD controller:
        h_peak <= Kd * theta_0 * sqrt(Kp / (2*J))

    where theta_0 is the initial attitude error.
    """
    theta_0 = sp.Symbol("theta_0", positive=True)
    omega_n = sp.sqrt(Kp / (2 * Jxx))
    h_peak = Kd * theta_0 * omega_n
    return h_peak


def evaluate_wheel_momentum(params: dict[str, float]) -> dict[str, float]:
    """Evaluate peak wheel momentum for a 10-degree slew."""
    import math
    Ixx_val, _, _ = evaluate_inertia(params)
    kp_val = params["Kp"]
    kd_val = params["Kd"]
    h_max_val = params["maxMomentum"]

    theta_0 = 10.0 * math.pi / 180  # 10 degree initial error
    omega_n = math.sqrt(kp_val / (2 * Ixx_val))
    h_peak = kd_val * theta_0 * omega_n

    return {
        "h_peak": h_peak,
        "h_max": h_max_val,
        "margin": h_max_val - h_peak,
        "theta_0_deg": 10.0,
    }


# ---------------------------------------------------------------------------
# Collected analysis results
# ---------------------------------------------------------------------------

@dataclass
class SymbolicAnalysisResult:
    """Collected results from all symbolic analyses."""
    inertia: tuple[float, float, float]
    eigenvalues: dict[str, list[complex]]
    stability_margins: dict[str, float]
    pointing_budget: dict[str, float]
    gravity_gradient: dict[str, float]
    wheel_momentum: dict[str, float]


def run_symbolic_analysis(params: dict[str, float]) -> SymbolicAnalysisResult:
    """Run all symbolic analyses and return collected results."""
    inertia = evaluate_inertia(params)
    eigs = evaluate_eigenvalues(params)
    margins = stability_margins(params)
    pointing = evaluate_pointing_budget(params)
    gg = evaluate_gravity_gradient(params)
    wheel = evaluate_wheel_momentum(params)

    return SymbolicAnalysisResult(
        inertia=inertia,
        eigenvalues=eigs,
        stability_margins=margins,
        pointing_budget=pointing,
        gravity_gradient=gg,
        wheel_momentum=wheel,
    )
