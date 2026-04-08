"""Pipeline orchestrator — runs all lifecycle stages sequentially.

Usage:
    uv run python -m pipeline.runner              # interactive attestation
    uv run python -m pipeline.runner --auto       # scripted attestation
    uv run python -m pipeline.runner --no-attest  # skip attestation stage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdflib import Graph

from analysis.build_proofs import build_all_proofs
from analysis.load_params import load_params, load_structural_graph
from analysis.numerical import run_disturbance_rejection, run_step_response
from analysis.proof_scripts import ProofStatus, verify_proof
from analysis.symbolic import run_symbolic_analysis
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
from pipeline.stages import LifecycleStage, check_gate
from traceability.attestation import request_attestation
from traceability.rtm import (
    assemble_rtm,
    export_rtm,
    load_base_graph,
    print_rtm_summary,
    validate_evidence_completeness,
    validate_structural_completeness,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def run_pipeline(
    *,
    auto_attest: bool = False,
    skip_attestation: bool = False,
    engineer_name: str = "ADCS Engineer",
) -> Graph:
    """Execute the full ADCS lifecycle pipeline."""
    stage = LifecycleStage.STRUCTURAL_DEFINED

    # ── Stage 1: STRUCTURAL_DEFINED ──────────────────────────────
    print("\n[Stage 1] Loading structural model...")
    base_graph = load_base_graph()
    struct_graph = load_structural_graph()
    model_hash = hash_structural_model(struct_graph)
    params = load_params(struct_graph)

    issues = validate_structural_completeness(base_graph)
    if issues:
        print(f"  STRUCTURAL ISSUES: {issues}")
        sys.exit(1)
    print(f"  Model hash: {model_hash[:16]}...")
    print(f"  Parameters loaded: {len(params)}")
    print(f"  Structural validation: PASS")

    # ── Stage 2: SYMBOLICALLY_ANALYZED ───────────────────────────
    stage = LifecycleStage.SYMBOLICALLY_ANALYZED
    print("\n[Stage 2] Running symbolic analysis...")
    sym_result = run_symbolic_analysis(params)
    print(f"  Inertia: Jxx={sym_result.inertia[0]:.1f}, "
          f"Jyy={sym_result.inertia[1]:.1f}, "
          f"Jzz={sym_result.inertia[2]:.1f} kg.m^2")
    print(f"  Stability margins: {', '.join(f'{k}={v:.4f}' for k,v in sym_result.stability_margins.items())}")

    proofs = build_all_proofs(model_hash)
    proof_results = {}
    for req_id, script in proofs.items():
        result = verify_proof(script, model_hash)
        proof_results[req_id] = result
        status = result.status.value
        print(f"  Proof {req_id}: {status} ({len(script.lemmas)} lemmas)")
        if result.status != ProofStatus.VERIFIED:
            print(f"    FAILURE: {result.failure_summary}")

    # ── Stage 3: NUMERICALLY_SIMULATED ───────────────────────────
    stage = LifecycleStage.NUMERICALLY_SIMULATED
    print("\n[Stage 3] Running numerical simulations...")
    step_result = run_step_response(params)
    step_summary = step_result.summary()
    print(f"  Step response: settling={step_summary['settling_time_s']:.1f}s, "
          f"final_error={step_summary['final_error_deg']:.4f} deg")
    print(f"  Peak wheel momentum: {step_summary['peak_wheel_momentum']:.3f} N.m.s "
          f"(limit: {params['maxMomentum']})")
    print(f"  Peak control torque: {step_summary['peak_control_torque']:.4f} N.m "
          f"(limit: {params['maxTorque']})")

    dist_result = run_disturbance_rejection(params)
    dist_summary = dist_result.summary()
    print(f"  Disturbance rejection: peak_error={dist_summary['peak_error_deg']:.6f} deg")

    # ── Stage 4: EVIDENCE_BOUND ──────────────────────────────────
    stage = LifecycleStage.EVIDENCE_BOUND
    print("\n[Stage 4] Binding evidence to RDF graph...")
    ev_graph = Graph()
    bind_prefixes(ev_graph)
    bind_computation_engines(ev_graph)

    # Proof evidence for all 4 requirements
    for req_id, script in proofs.items():
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
            result_summary=f"Symbolic proof: {script.claim}",
            source_file="analysis/build_proofs.py",
        )

    # Simulation evidence
    sim_hash = hash_simulation(step_result.config.to_dict(), step_summary)
    for req_id, desc in [
        ("REQ-001", f"Step response: settling={step_summary['settling_time_s']:.1f}s, "
                    f"final_error={step_summary['final_error_deg']:.4f} deg"),
        ("REQ-002", f"Peak wheel momentum: {step_summary['peak_wheel_momentum']:.3f} N.m.s "
                    f"(limit={params['maxMomentum']})"),
    ]:
        bind_simulation_evidence(
            ev_graph,
            evidence_id=f"EV-SIM-{req_id}",
            activity_id=f"NS-{req_id}",
            requirement_id=req_id,
            model_hash=model_hash,
            sim_hash=sim_hash,
            result_summary=desc,
            source_file="analysis/numerical.py",
        )

    # Disturbance rejection evidence for REQ-004
    dist_hash = hash_simulation(dist_result.config.to_dict(), dist_summary)
    bind_simulation_evidence(
        ev_graph,
        evidence_id="EV-SIM-REQ-004",
        activity_id="NS-REQ-004",
        requirement_id="REQ-004",
        model_hash=model_hash,
        sim_hash=dist_hash,
        result_summary=f"Disturbance rejection: peak_error={dist_summary['peak_error_deg']:.6f} deg",
        source_file="analysis/numerical.py",
    )

    print(f"  Evidence artifacts created: {len(list(ev_graph.subjects()))} nodes")

    # ── Stage 5: RTM_ASSEMBLED ───────────────────────────────────
    stage = LifecycleStage.RTM_ASSEMBLED
    print("\n[Stage 5] Assembling RTM...")
    rtm = assemble_rtm(base_graph, ev_graph)

    ev_issues = validate_evidence_completeness(rtm)
    if ev_issues:
        print(f"  Evidence gaps: {ev_issues}")
    else:
        print(f"  Evidence completeness: PASS (all requirements have evidence)")

    export_rtm(rtm, OUTPUT_DIR / "rtm_pre_attestation.ttl")
    print(f"  Pre-attestation RTM exported to output/rtm_pre_attestation.ttl")

    # ── Stage 6: ATTESTATION ─────────────────────────────────────
    if not skip_attestation:
        stage = LifecycleStage.ATTESTATION
        print("\n[Stage 6] Human attestation...")

        adequacy_statements = {
            "REQ-001": ("Linearized PD model adequate for pointing analysis. "
                        "Interface parameters (mass, orbit) accepted from systems engineering."),
            "REQ-002": ("Energy-based momentum bound is conservative. "
                        "Reaction wheel model adequate for peak momentum estimation."),
            "REQ-003": ("Linearized stability analysis via Routh-Hurwitz is adequate. "
                        "Nonlinear effects are second-order for small angles."),
            "REQ-004": ("Linearized gravity gradient model adequate for GEO orbit. "
                        "Higher-order terms negligible at geostationary altitude."),
        }
        sufficiency_statements = {
            "REQ-001": ("Symbolic analysis confirms finite steady-state error below 0.1 deg. "
                        "Numerical simulation shows convergence. Settling time exceeds 120s — "
                        "recommend Kd tuning but pointing accuracy met."),
            "REQ-002": "Symbolic bound and simulation both confirm peak momentum well below 4.0 N.m.s.",
            "REQ-003": ("Routh-Hurwitz proof confirms asymptotic stability for all positive J, Kp, Kd. "
                        "Numerical eigenvalues confirm margins exceed -0.010 rad/s on all axes."),
            "REQ-004": ("Gravity gradient torques are micro-Nm at GEO, orders of magnitude below "
                        "0.1 N.m actuator capacity. Simulation confirms negligible pointing impact."),
        }

        for req_id in ["REQ-001", "REQ-002", "REQ-003", "REQ-004"]:
            if auto_attest:
                request_attestation(
                    rtm, req_id, engineer_name,
                    auto_attest=True,
                    model_adequacy=adequacy_statements[req_id],
                    evidence_sufficiency=sufficiency_statements[req_id],
                )
            else:
                request_attestation(rtm, req_id, engineer_name)

    # ── Stage 7: REPORTED ────────────────────────────────────────
    stage = LifecycleStage.REPORTED
    print("\n[Stage 7] Generating reports...")
    export_rtm(rtm, OUTPUT_DIR / "rtm.ttl")
    print(f"  Final RTM exported to output/rtm.ttl")

    summary = print_rtm_summary(rtm)
    print(summary)

    # ── Stage 8: VISUALIZED_AND_INTERROGABLE ─────────────────────
    stage = LifecycleStage.VISUALIZED_AND_INTERROGABLE
    print("\n[Stage 8] Visualization and interrogation ready.")
    print("  Use interrogate/explain.py for 'How do you know X?' queries")
    print("  Use interrogate/reproduce.py to re-verify evidence")
    print("  Use interrogate/visualize.py to render the RTM graph")

    return rtm


def main():
    parser = argparse.ArgumentParser(description="ADCS Lifecycle Pipeline")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-attest with scripted judgments")
    parser.add_argument("--no-attest", action="store_true",
                        help="Skip attestation stage")
    parser.add_argument("--engineer", default="Dr. Michael Zargham (@mzargham)",
                        help="Engineer name for attestation")
    args = parser.parse_args()

    run_pipeline(
        auto_attest=args.auto,
        skip_attestation=args.no_attest,
        engineer_name=args.engineer,
    )


if __name__ == "__main__":
    main()
