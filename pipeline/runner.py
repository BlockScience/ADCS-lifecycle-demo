"""Pipeline orchestrator — runs all lifecycle stages sequentially.

`run_pipeline` threads a `PipelineState` object through per-stage
free functions; each stage returns a typed result record assigned to
the matching state field. The orchestrator's job is narration + the
ordered call sequence. Stage bodies are in this module as
`run_stage_<N>_<name>(state)` functions; future WPs can re-import
them from `pipeline.runner` without going through the CLI.

Usage:
    uv run python -m pipeline.runner              # interactive attestation
    uv run python -m pipeline.runner --auto       # scripted attestation
    uv run python -m pipeline.runner --no-attest  # skip attestation stage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdflib import Dataset

from analysis.build_proofs import build_all_proofs
from analysis.load_params import load_params, load_structural_graph
from analysis.proof_scripts import ProofStatus, verify_proof
from compute import get_compute_backend
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
from pipeline.backends import get_backend
from pipeline.dataset import graph_for, load_into
from pipeline.stage0_assembly import run_stage_0
from pipeline.state import (
    AttestationStageResult,
    AuditStageResult,
    ClosureRuleResult,
    EvidenceBindingResult,
    NumericalResult,
    PipelineState,
    ReportStageResult,
    StructuralResult,
    SymbolicResult,
)
from traceability.attestation import OUTCOME_FAILED, request_attestation
from traceability.plan_execution import emit_stage_activity
from traceability.rtm import (
    STRUCTURAL_DIR,
    export_rtm,
    print_rtm_summary,
    validate_evidence_completeness,
    validate_structural_completeness,
)
from traceability.audit import audit as run_audit, emit_audit_graph, render_report
from traceability.validation import validate as validate_closure_rules

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


# ── Stage 1: STRUCTURAL_DEFINED ──────────────────────────────────────
def run_stage_1_structural(state: PipelineState) -> StructuralResult:
    emit_stage_activity(state.ds, "LoadStructural")
    print("\n[Stage 1] Loading structural model...")
    triples_loaded = 0
    for ttl in sorted(STRUCTURAL_DIR.glob("*.ttl")):
        triples_loaded += load_into(state.ds, "structural", ttl)
    struct_graph = load_structural_graph()
    model_hash = hash_structural_model(struct_graph)
    params = load_params(struct_graph)

    issues = validate_structural_completeness(state.ds)
    if issues:
        print(f"  STRUCTURAL ISSUES: {issues}")
        sys.exit(1)
    print(f"  Model hash: {model_hash[:16]}...")
    print(f"  Parameters loaded: {len(params)}")
    print(f"  Structural validation: PASS")

    print(f"\n  Compute: {state.compute_backend.describe()}")
    return StructuralResult(
        model_hash=model_hash, params=params, triples_loaded=triples_loaded,
    )


# ── Stage 2: SYMBOLICALLY_ANALYZED ───────────────────────────────────
def run_stage_2_symbolic(state: PipelineState) -> SymbolicResult:
    assert state.structural is not None, "Stage 2 requires Stage 1 result"
    emit_stage_activity(state.ds, "SymbolicAnalysis")
    print("\n[Stage 2] Running symbolic analysis...")
    sym_result, sym_meta = state.compute_backend.run_symbolic_analysis(
        state.structural.params,
    )
    print(f"  Inertia: Jxx={sym_result.inertia[0]:.1f}, "
          f"Jyy={sym_result.inertia[1]:.1f}, "
          f"Jzz={sym_result.inertia[2]:.1f} kg.m^2")
    print(f"  Stability margins: "
          f"{', '.join(f'{k}={v:.4f}' for k, v in sym_result.stability_margins.items())}")

    proofs = build_all_proofs(state.structural.model_hash)
    proof_results: dict[str, object] = {}
    for req_id, script in proofs.items():
        result = verify_proof(script, state.structural.model_hash)
        proof_results[req_id] = result
        print(f"  Proof {req_id}: {result.status.value} ({len(script.lemmas)} lemmas)")
        if result.status != ProofStatus.VERIFIED:
            print(f"    FAILURE: {result.failure_summary}")

    return SymbolicResult(
        sym_result=sym_result, sym_meta=sym_meta,
        proofs=proofs, proof_results=proof_results,
    )


# ── Stage 3: NUMERICALLY_SIMULATED ───────────────────────────────────
def run_stage_3_numerical(state: PipelineState) -> NumericalResult:
    assert state.structural is not None, "Stage 3 requires Stage 1 result"
    emit_stage_activity(state.ds, "NumericalSimulation")
    print("\n[Stage 3] Running numerical simulations...")
    params = state.structural.params

    step_result, step_meta = state.compute_backend.run_step_response(params)
    step_summary = step_result.summary()
    print(f"  Step response: settling={step_summary['settling_time_s']:.1f}s, "
          f"final_error={step_summary['final_error_deg']:.4f} deg")
    print(f"  Peak wheel momentum: {step_summary['peak_wheel_momentum']:.3f} N.m.s "
          f"(limit: {params['maxMomentum']})")
    print(f"  Peak control torque: {step_summary['peak_control_torque']:.4f} N.m "
          f"(limit: {params['maxTorque']})")

    dist_result, dist_meta = state.compute_backend.run_disturbance_rejection(params)
    dist_summary = dist_result.summary()
    if state.compute_name != "local":
        print(f"  Execution captured: host={step_meta.hostname}, "
              f"image={step_meta.image_label or 'n/a'}, "
              f"container={step_meta.container_id or 'n/a'}")
    print(f"  Disturbance rejection: peak_error={dist_summary['peak_error_deg']:.6f} deg")

    return NumericalResult(
        step_result=step_result, step_meta=step_meta, step_summary=step_summary,
        dist_result=dist_result, dist_meta=dist_meta, dist_summary=dist_summary,
    )


# ── Stage 4: EVIDENCE_BOUND ──────────────────────────────────────────
def run_stage_4_bind_evidence(state: PipelineState) -> EvidenceBindingResult:
    assert state.structural is not None, "Stage 4 requires Stage 1 result"
    assert state.symbolic is not None, "Stage 4 requires Stage 2 result"
    assert state.numerical is not None, "Stage 4 requires Stage 3 result"

    emit_stage_activity(state.ds, "BindEvidence")
    print("\n[Stage 4] Binding evidence to RDF graph...")
    ev_graph = graph_for(state.ds, "evidence")
    bind_computation_engines(ev_graph)

    params = state.structural.params
    model_hash = state.structural.model_hash
    proofs = state.symbolic.proofs
    sym_meta = state.symbolic.sym_meta
    step_result = state.numerical.step_result
    step_summary = state.numerical.step_summary
    step_meta = state.numerical.step_meta
    dist_result = state.numerical.dist_result
    dist_summary = state.numerical.dist_summary
    dist_meta = state.numerical.dist_meta

    # Proof evidence for all 4 requirements.
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
            execution_metadata=sym_meta,
        )

    # Simulation evidence for REQ-001, REQ-002.
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
            execution_metadata=step_meta,
        )

    # Disturbance rejection evidence for REQ-004.
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
        execution_metadata=dist_meta,
    )

    evidence_node_count = len(list(ev_graph.subjects()))
    print(f"  Evidence artifacts created: {evidence_node_count} nodes "
          f"(written to <adcs:evidence>)")
    return EvidenceBindingResult(evidence_node_count=evidence_node_count)


# ── Stage 5: RTM_ASSEMBLED ───────────────────────────────────────────
def run_stage_5_assemble_rtm(state: PipelineState) -> None:
    emit_stage_activity(state.ds, "AssembleRTM")
    print("\n[Stage 5] Assembling RTM...")
    # The Dataset already contains structural + ontology + evidence in
    # their respective named graphs; assembly is a no-op for the runtime
    # path (default_union exposes the merged view to queries).
    ev_issues = validate_evidence_completeness(state.ds)
    if ev_issues:
        print(f"  Evidence gaps: {ev_issues}")
    else:
        print(f"  Evidence completeness: PASS (all requirements have evidence)")

    export_rtm(state.ds, OUTPUT_DIR / "rtm_pre_attestation.ttl")
    print(f"  Pre-attestation RTM exported to output/rtm_pre_attestation.{{ttl,trig}}")


# ── Stage 6: ATTESTATION ─────────────────────────────────────────────
def run_stage_6_attestation(state: PipelineState) -> AttestationStageResult:
    if state.skip_attestation:
        return AttestationStageResult(attestation_uris=None)

    assert state.structural is not None and state.numerical is not None, (
        "Stage 6 requires Stages 1 + 3"
    )
    emit_stage_activity(state.ds, "Attest")
    print("\n[Stage 6] Human attestation...")

    params = state.structural.params
    step_summary = state.numerical.step_summary

    adequacy_statements = {
        "REQ-002": ("Energy-based momentum bound is conservative. "
                    "Reaction wheel model adequate for peak momentum estimation."),
        "REQ-003": ("Linearized stability analysis via Routh-Hurwitz is adequate. "
                    "Nonlinear effects are second-order for small angles."),
        "REQ-004": ("Linearized gravity gradient model adequate for GEO orbit. "
                    "Higher-order terms negligible at geostationary altitude."),
    }
    sufficiency_statements = {
        "REQ-002": "Symbolic bound and simulation both confirm peak momentum well below 4.0 N.m.s.",
        "REQ-003": ("Routh-Hurwitz proof confirms asymptotic stability for all positive J, Kp, Kd. "
                    "Numerical eigenvalues confirm margins exceed -0.010 rad/s on all axes."),
        "REQ-004": ("Gravity gradient torques are micro-Nm at GEO, orders of magnitude below "
                    "0.1 N.m actuator capacity. Simulation confirms negligible pointing impact."),
    }

    print(f"\n  REQ-001: ATTESTATION DECLINED")
    print(f"    Settling time {step_summary['settling_time_s']:.0f}s exceeds 120s requirement.")
    print(f"    Action: retune gains (Kp: {params['Kp']:.0f}->4, Kd: {params['Kd']:.0f}->30) and re-verify.")
    if state.auto_attest:
        request_attestation(
            state.ds, "REQ-001", state.engineer_name,
            auto_attest=True,
            model_adequacy=(
                "Step-response simulation is adequate for evaluating pointing-"
                "accuracy settling time at this point in the lifecycle."
            ),
            evidence_sufficiency=(
                f"Evidence is sufficient to conclude REQ-001 is NOT yet satisfied: "
                f"settling time {step_summary['settling_time_s']:.0f}s exceeds the "
                f"120s requirement. Action item: retune gains "
                f"(Kp: {params['Kp']:.0f}->4, Kd: {params['Kd']:.0f}->30) and re-verify."
            ),
            outcome=OUTCOME_FAILED,
        )

    for req_id in ["REQ-002", "REQ-003", "REQ-004"]:
        if state.auto_attest:
            request_attestation(
                state.ds, req_id, state.engineer_name,
                auto_attest=True,
                model_adequacy=adequacy_statements[req_id],
                evidence_sufficiency=sufficiency_statements[req_id],
            )
        else:
            request_attestation(state.ds, req_id, state.engineer_name)

    return AttestationStageResult(attestation_uris=None)


# ── Stage 6.5: VALIDATE CLOSURE RULES ────────────────────────────────
def run_stage_6_5_validate_closure(state: PipelineState) -> ClosureRuleResult:
    emit_stage_activity(state.ds, "ValidateShapes")
    print("\n[Stage 6.5] Validating closure-rule suite...")
    report = validate_closure_rules(state.ds, skip_reverification=False)
    for line in report.summary_lines():
        print(f"  {line}")
    # Violations are surfaced but do not fail the pipeline by default —
    # the audit module renders a structured report. CI can opt into
    # hard-fail behaviour by checking `report.conforms`.
    return ClosureRuleResult(report=report)


# ── Stage 7a: AUDIT TRACE ────────────────────────────────────────────
def run_stage_7a_audit(state: PipelineState) -> AuditStageResult:
    emit_stage_activity(state.ds, "AuditTrace")
    print("\n[Stage 7a] Auditing forward / backward / bidirectional traceability...")
    audit_report = run_audit(state.ds)
    print(f"  {audit_report.forward.summary()}")
    print(f"  {audit_report.backward.summary()}")
    bidirect = audit_report.bidirectional()
    print(f"  Bidirectional: {'PASS' if bidirect.passed else 'FAIL'}")
    if audit_report.orphans.any:
        if audit_report.orphans.requirements_without_evidence:
            print(f"  Orphan reqs (no evidence): "
                  f"{audit_report.orphans.requirements_without_evidence}")
        if audit_report.orphans.evidence_without_requirement:
            print(f"  Orphan evidence: {len(audit_report.orphans.evidence_without_requirement)}")
        if audit_report.orphans.attestations_with_broken_refs:
            print(f"  Broken attestations: {len(audit_report.orphans.attestations_with_broken_refs)}")
    else:
        print("  Orphans: none")
    emit_audit_graph(state.ds, audit_report)

    audit_md = OUTPUT_DIR / "audit.md"
    audit_md.parent.mkdir(parents=True, exist_ok=True)
    audit_md.write_text(render_report(audit_report, fmt="md"))
    (OUTPUT_DIR / "audit.csv").write_text(render_report(audit_report, fmt="csv"))
    print(f"  Audit report -> output/audit.md, output/audit.csv")

    return AuditStageResult(report=audit_report)


# ── Stage 7: REPORTED ────────────────────────────────────────────────
def run_stage_7_report(state: PipelineState) -> ReportStageResult:
    emit_stage_activity(state.ds, "Report")
    print("\n[Stage 7] Generating reports...")
    store = get_backend(state.backend_name)
    print(f"  Backend: {store.describe()}")
    persisted = store.persist(state.ds, OUTPUT_DIR)
    print(f"  Persisted {len(persisted)} named graphs "
          f"({sum(persisted.values())} triples total)")

    summary = print_rtm_summary(state.ds)
    print(summary)
    return ReportStageResult(persisted_graphs=persisted, backend_name=state.backend_name)


# ── Stage 8: VISUALIZED_AND_INTERROGABLE ─────────────────────────────
def run_stage_8_interrogate(state: PipelineState) -> None:
    emit_stage_activity(state.ds, "Interrogate")
    print("\n[Stage 8] Visualization and interrogation ready.")
    print("  Use interrogate/explain.py for 'How do you know X?' queries")
    print("  Use interrogate/reproduce.py to re-verify evidence")
    print("  Use interrogate/visualize.py to render the RTM graph")


# ── Orchestrator ─────────────────────────────────────────────────────
def run_pipeline(
    *,
    auto_attest: bool = False,
    skip_attestation: bool = False,
    engineer_name: str = "ADCS Engineer",
    rebuild_ontology: bool = False,
    backend: str = "local",
    compute: str = "local",
) -> Dataset:
    """Execute the full ADCS lifecycle pipeline.

    Returns the populated Dataset with eight named graphs:
        <rtm:ontology>, <rtm:plan>, <adcs:structural>, <adcs:context>,
        <adcs:evidence>, <adcs:attestations>, <adcs:plan-execution>,
        <adcs:audit>.
    Default-union is enabled so consumers can query across the union
    with plain SPARQL.
    """
    ds = run_stage_0(rebuild=rebuild_ontology)
    state = PipelineState(
        ds=ds,
        compute_backend=get_compute_backend(compute),
        engineer_name=engineer_name,
        auto_attest=auto_attest,
        skip_attestation=skip_attestation,
        backend_name=backend,
        compute_name=compute,
    )
    state.structural    = run_stage_1_structural(state)
    state.symbolic      = run_stage_2_symbolic(state)
    state.numerical     = run_stage_3_numerical(state)
    state.evidence      = run_stage_4_bind_evidence(state)
    run_stage_5_assemble_rtm(state)
    state.attestation   = run_stage_6_attestation(state)
    state.closure_rules = run_stage_6_5_validate_closure(state)
    state.audit         = run_stage_7a_audit(state)
    state.report        = run_stage_7_report(state)
    run_stage_8_interrogate(state)
    return state.ds


def main():
    parser = argparse.ArgumentParser(description="ADCS Lifecycle Pipeline")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-attest with scripted judgments")
    parser.add_argument("--no-attest", action="store_true",
                        help="Skip attestation stage")
    parser.add_argument("--engineer", default="Dr. Michael Zargham (@mzargham)",
                        help="Engineer name for attestation")
    parser.add_argument("--rebuild", action="store_true",
                        help="Invoke `make ontology` before Stage 0 (live-demo rebuild path)")
    parser.add_argument("--backend", choices=["local", "flexo", "fuseki"], default="local",
                        help="Persistence backend (default: local filesystem)")
    parser.add_argument("--compute", choices=["local", "docker"], default="local",
                        help="Compute backend for Stage 2/3 analysis. "
                             "`docker` emulates remote compute and captures "
                             "image/hostname/container-ID as RTM provenance.")
    args = parser.parse_args()

    run_pipeline(
        auto_attest=args.auto,
        skip_attestation=args.no_attest,
        engineer_name=args.engineer,
        rebuild_ontology=args.rebuild,
        backend=args.backend,
        compute=args.compute,
    )


if __name__ == "__main__":
    main()
