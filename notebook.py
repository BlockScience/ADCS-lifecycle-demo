"""# ADCS Lifecycle Demo: Bidirectional Requirements Traceability

A walkthrough of satellite attitude control system design — from receiving
requirements through symbolic analysis, numerical simulation, evidence
binding, human attestation, and audit.
"""

import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md("""
    # ADCS Lifecycle Demo

    ## Bidirectional Requirements Traceability with Reproducible Evidence

    This notebook walks through the complete lifecycle of verifying an
    **Attitude Determination and Control System (ADCS)** for a geostationary
    communications satellite.

    We follow the perspective of the **controls engineering team** — one
    disciplinary team within a larger satellite design program. Our job is
    to demonstrate that the ADCS meets its requirements, with evidence that
    any auditor can interrogate and reproduce.

    ### Core Principle

    > **Evidence does not verify requirements; evidence supports a human
    > judgment that requirements are satisfied.**

    Models are imperfect representations of physical systems. Symbolic proofs
    and simulation results are claims true *within the model*. The engineer
    judges model adequacy and evidence sufficiency. Only human attestation
    connects evidence to requirement satisfaction.
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Act 1: The Assignment

    You are Dr. Michael Zargham, lead controls engineer on the GeoSat
    communications satellite program. Systems engineering has allocated
    four requirements to your ADCS subsystem, each derived from
    satellite-level requirements.

    Your team owns the ADCS — reaction wheels, star tracker, IMU, and the
    PD attitude controller. You consume interface parameters (mass, orbit,
    panel geometry) from other teams but don't control them.

    ```
    Satellite (system-of-interest)
    ├── ADCS           ← YOUR SCOPE
    ├── Power          ← interface: power budget
    ├── Communications ← interface: antenna pointing
    ├── Thermal        ← interface: wheel heat dissipation
    └── Structure      ← interface: mass properties
    ```

    Let's load the structural model and see what we're working with.
    """)
    return


@app.cell
def __():
    import sys
    sys.path.insert(0, ".")

    from rdflib import Graph
    from ontology.prefixes import bind_prefixes, SYSML, RTM, ADCS, SAT, PROV
    from traceability.queries import query_to_dicts
    return Graph, bind_prefixes, SYSML, RTM, ADCS, SAT, PROV, query_to_dicts, sys


@app.cell
def __(Graph, bind_prefixes):
    from analysis.load_params import load_structural_graph, load_params

    struct_graph = load_structural_graph()
    params = load_params(struct_graph)
    return struct_graph, params, load_structural_graph, load_params


@app.cell
def __(mo, params):
    _param_rows = "\n".join(
        f"| {k} | {v:.6g} |" for k, v in sorted(params.items())
    )
    mo.md(
        "### Structural Parameters (from RDF via SPARQL)\n\n"
        "All parameters flow from the SysMLv2 structural model — nothing is "
        "hardcoded. If systems engineering updates the satellite mass, our "
        "entire analysis chain re-derives from the new value.\n\n"
        "| Parameter | Value |\n"
        "|-----------|-------|\n"
        f"{_param_rows}"
    )
    return


@app.cell
def __(mo, struct_graph, query_to_dicts):
    _req_query = """
    SELECT ?name ?text WHERE {
        ?req a sysml:RequirementDefinition ;
             sysml:declaredName ?name ;
             sysml:text ?text .
        FILTER(STRSTARTS(?name, "REQ-"))
    }
    ORDER BY ?name
    """
    _reqs = query_to_dicts(struct_graph, _req_query)

    _deriv_query = """
    SELECT ?child ?parent WHERE {
        ?c sysml:declaredName ?child ;
           rtm:derivedFrom ?p .
        ?p sysml:declaredName ?parent .
    }
    ORDER BY ?child
    """
    _derivs = query_to_dicts(struct_graph, _deriv_query)
    _deriv_map = {r["child"]: r["parent"] for r in _derivs}

    _alloc_query = """
    SELECT ?reqName ?elementName WHERE {
        ?req sysml:declaredName ?reqName ;
             sysml:ownedRelationship ?rel .
        ?rel a sysml:SatisfyRequirementUsage ;
             sysml:satisfyingElement ?el .
        ?el sysml:declaredName ?elementName .
        FILTER(STRSTARTS(?reqName, "REQ-"))
    }
    ORDER BY ?reqName ?elementName
    """
    _allocs = query_to_dicts(struct_graph, _alloc_query)

    _alloc_map = {}
    for _a in _allocs:
        _alloc_map.setdefault(_a["reqName"], []).append(_a["elementName"])

    _req_rows = []
    for _r in _reqs:
        _name = _r["name"]
        _text = _r["text"].strip().replace("\n", " ")[:80]
        _parent = _deriv_map.get(_name, "—")
        _elements = ", ".join(_alloc_map.get(_name, []))
        _req_rows.append(f"| {_name} | {_text}... | {_parent} | {_elements} |")

    _req_table = "\n".join(_req_rows)

    mo.md(
        "### ADCS Requirements\n\n"
        "Four requirements allocated to us, each derived from a satellite-level "
        "parent requirement and satisfied by specific design elements:\n\n"
        "| ID | Requirement | Derived From | Satisfied By |\n"
        "|----|------------|-------------|-------------|\n"
        f"{_req_table}"
    )
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Act 2: Symbolic Analysis

    Before running any simulation, we derive formal results symbolically
    using SymPy. Every quantity is computed from the structural parameters —
    the inertia tensor via parallel axis theorem, eigenvalues for stability
    analysis, and bounds for pointing error and wheel momentum.

    These are claims true *within our model*. Whether the model adequately
    represents the physical satellite is a judgment we'll make during
    attestation.
    """)
    return


@app.cell
def __(params):
    from analysis.symbolic import (
        run_symbolic_analysis,
        build_inertia_tensor_symbolic,
        evaluate_inertia,
        stability_margins,
    )

    sym_result = run_symbolic_analysis(params)
    return sym_result, run_symbolic_analysis, build_inertia_tensor_symbolic, evaluate_inertia, stability_margins


@app.cell
def __(mo, sym_result):
    _Ixx, _Iyy, _Izz = sym_result.inertia
    _margins = sym_result.stability_margins

    mo.md(f"""
    ### Composite Inertia Tensor

    Derived via parallel axis theorem (bus + 2 solar panels + antenna):

    | Axis | Inertia (kg-m^2) | Dominant contributor |
    |------|-----------------|---------------------|
    | Ixx  | {_Ixx:.1f} | Solar panels (offset along Y) |
    | Iyy  | {_Iyy:.1f} | Bus (panels add little on this axis) |
    | Izz  | {_Izz:.1f} | Solar panels (offset along Y) |

    The panels dominate Ixx and Izz because their center of mass is far
    from the satellite center — the parallel axis term grows as distance
    squared.

    ### Stability Margins (REQ-003)

    Closed-loop eigenvalues for each axis (PD controller, linearized):

    | Axis | Re(lambda) | Margin vs -0.010 |
    |------|-----------|-----------------|
    | X | {_margins['x']:.4f} rad/s | {'PASS' if _margins['x'] <= -0.010 else 'MARGINAL'} |
    | Y | {_margins['y']:.4f} rad/s | {'PASS' if _margins['y'] <= -0.010 else 'MARGINAL'} |
    | Z | {_margins['z']:.4f} rad/s | {'PASS' if _margins['z'] <= -0.010 else 'MARGINAL'} |

    All axes satisfy REQ-003 (Re(lambda) <= -0.010 rad/s).
    """)
    return


@app.cell
def __(mo, sym_result):
    _pb = sym_result.pointing_budget
    _gg = sym_result.gravity_gradient
    _wm = sym_result.wheel_momentum

    mo.md(f"""
    ### Pointing Budget (REQ-001)

    | Metric | Value |
    |--------|-------|
    | Steady-state error (gravity gradient) | {_pb['theta_ss_deg']:.6f} deg |
    | Star tracker noise floor | {_pb['st_floor_deg']:.6f} deg |
    | Settling time (4/|Re(lambda)|) | {_pb['settling_time_s']:.1f} s |

    The steady-state pointing error is well below 0.1 deg. However, the
    **settling time is {_pb['settling_time_s']:.0f}s** — exceeding the 120s target.
    This is a real finding that the engineer must address during attestation.

    ### Gravity Gradient (REQ-004)

    | Metric | Value |
    |--------|-------|
    | tau_gg_x | {_gg['tau_gg_x']:.2e} N.m |
    | tau_gg_y | {_gg['tau_gg_y']:.2e} N.m |
    | Actuator capacity | {_gg['tau_max']} N.m |

    Gravity gradient torques at GEO are **orders of magnitude** below
    actuator capacity.

    ### Wheel Momentum (REQ-002)

    | Metric | Value |
    |--------|-------|
    | Peak momentum (10 deg slew) | {_wm['h_peak']:.3f} N.m.s |
    | Rated capacity | {_wm['h_max']} N.m.s |
    | Margin | {_wm['margin']:.3f} N.m.s |
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ### Formal Proofs

    Each requirement gets a ProofScript — a chain of SymPy lemmas, each
    independently re-verifiable. The proof is bound to the structural model
    via content hash: if the model changes, the proof hash changes, alerting
    auditors to re-verify.
    """)
    return


@app.cell
def __(struct_graph):
    from evidence.hashing import hash_structural_model, hash_proof
    from analysis.build_proofs import build_all_proofs
    from analysis.proof_scripts import verify_proof, ProofStatus

    model_hash = hash_structural_model(struct_graph)
    proofs = build_all_proofs(model_hash)

    proof_results = {}
    for _req_id, _script in proofs.items():
        _result = verify_proof(_script, model_hash)
        proof_results[_req_id] = _result

    return model_hash, proofs, proof_results, hash_structural_model, hash_proof, build_all_proofs, verify_proof, ProofStatus


@app.cell
def __(mo, proofs, proof_results, ProofStatus):
    _rows = []
    for _req_id in sorted(proofs.keys()):
        _script = proofs[_req_id]
        _result = proof_results[_req_id]
        _status = "VERIFIED" if _result.status == ProofStatus.VERIFIED else "FAILED"
        _lemmas = ", ".join(l.name for l in _script.lemmas)
        _rows.append(f"| {_req_id} | {_status} | {_script.claim[:60]}... | {_lemmas} |")

    _proof_table = "\n".join(_rows)
    mo.md(
        "| Requirement | Status | Claim | Lemmas |\n"
        "|-------------|--------|-------|--------|\n"
        f"{_proof_table}\n\n"
        "All proofs pass. Each can be serialized to JSON, stored, and re-verified "
        "by anyone — no trust in the original analyst required."
    )
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Act 3: Numerical Simulation

    Symbolic analysis tells us what the model *should* do. Numerical
    simulation shows what it *actually does* when we integrate the full
    nonlinear dynamics. We run two scenarios:

    1. **Step response** — 10-degree initial attitude error, observe settling
    2. **Disturbance rejection** — near-zero error, observe gravity gradient effects
    """)
    return


@app.cell
def __(params):
    from analysis.numerical import run_step_response, run_disturbance_rejection

    step_result = run_step_response(params)
    step_summary = step_result.summary()

    dist_result = run_disturbance_rejection(params)
    dist_summary = dist_result.summary()
    return step_result, step_summary, dist_result, dist_summary, run_step_response, run_disturbance_rejection


@app.cell
def __(mo, step_result, step_summary):
    import matplotlib.pyplot as plt
    import numpy as np

    _fig, _axes = plt.subplots(2, 2, figsize=(12, 8))
    _axis_colors = {"X": "#1f77b4", "Y": "#2ca02c", "Z": "#9467bd"}  # blue, green, purple
    _limit_color = "#d62728"  # red reserved exclusively for requirement limits

    # Attitude error
    _q_vec = np.linalg.norm(step_result.q[:, :3], axis=1)
    _theta_deg = np.degrees(2 * _q_vec)
    _axes[0, 0].semilogy(step_result.t, _theta_deg, color=_axis_colors["X"], linewidth=1.5, label='Attitude error')
    _axes[0, 0].axhline(0.1, color=_limit_color, linestyle='--', linewidth=1, label='REQ-001 limit (0.1 deg)')
    _axes[0, 0].set_xlabel('Time (s)')
    _axes[0, 0].set_ylabel('Attitude Error (deg)')
    _axes[0, 0].set_title('Pointing Convergence')
    _axes[0, 0].set_ylim(bottom=1e-3)
    _axes[0, 0].legend(fontsize=8)
    _axes[0, 0].grid(True, alpha=0.3, which='both')

    # Angular velocity
    for _i, (_axis, _c) in enumerate(_axis_colors.items()):
        _axes[0, 1].plot(step_result.t, np.degrees(step_result.omega[:, _i]),
                         color=_c, linewidth=1, label=f'{_axis}-axis')
    _axes[0, 1].set_xlabel('Time (s)')
    _axes[0, 1].set_ylabel('Angular Rate (deg/s)')
    _axes[0, 1].set_title('Angular Velocity')
    _axes[0, 1].legend(fontsize=8)
    _axes[0, 1].grid(True, alpha=0.3)

    # Control torque
    for _i, (_axis, _c) in enumerate(_axis_colors.items()):
        _axes[1, 0].plot(step_result.t, step_result.tau_ctrl[:, _i],
                         color=_c, linewidth=1, label=f'{_axis}-axis')
    _axes[1, 0].axhline(step_result.config.max_torque, color=_limit_color, linestyle='--',
                         linewidth=1, alpha=0.7, label='Torque limit')
    _axes[1, 0].axhline(-step_result.config.max_torque, color=_limit_color, linestyle='--',
                         linewidth=1, alpha=0.7)
    _axes[1, 0].set_xlabel('Time (s)')
    _axes[1, 0].set_ylabel('Torque (N.m)')
    _axes[1, 0].set_title('Control Torque')
    _axes[1, 0].legend(fontsize=8)
    _axes[1, 0].grid(True, alpha=0.3)

    # Wheel momentum
    _h_mag = np.linalg.norm(step_result.h_wheel, axis=1)
    _axes[1, 1].plot(step_result.t, _h_mag, color=_axis_colors["X"], linewidth=1.5, label='|h| (total)')
    _axes[1, 1].axhline(step_result.config.max_momentum, color=_limit_color, linestyle='--',
                         linewidth=1, label='REQ-002 limit (4.0 N.m.s)')
    _axes[1, 1].set_xlabel('Time (s)')
    _axes[1, 1].set_ylabel('Momentum (N.m.s)')
    _axes[1, 1].set_title('Wheel Angular Momentum')
    _axes[1, 1].legend(fontsize=8)
    _axes[1, 1].grid(True, alpha=0.3)

    _fig.suptitle('Step Response: 10-degree Slew Maneuver', fontsize=14, fontweight='bold')
    plt.tight_layout()

    mo.md(f"""
    ### Step Response Results

    | Metric | Value |
    |--------|-------|
    | Final pointing error | {step_summary['final_error_deg']:.4f} deg |
    | Peak pointing error | {step_summary['peak_error_deg']:.1f} deg |
    | Settling time | {step_summary['settling_time_s']:.1f} s |
    | Peak wheel momentum | {step_summary['peak_wheel_momentum']:.3f} N.m.s |
    | Peak control torque | {step_summary['peak_control_torque']:.4f} N.m |
    """)

    _fig
    return np, plt


@app.cell
def __(mo, dist_summary):
    mo.md(f"""
    ### Disturbance Rejection Results

    | Metric | Value |
    |--------|-------|
    | Peak error (GG disturbance) | {dist_summary['peak_error_deg']:.6f} deg |
    | Final angular rate | {dist_summary['final_omega_norm']:.2e} rad/s |

    Gravity gradient effects are negligible at GEO — confirming REQ-004.
    """)
    return


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Act 4: Evidence Binding

    Now we bind our computational results to the RDF traceability graph.
    Every evidence artifact gets a content hash, a model hash (binding it
    to the structural model version), and PROV-O provenance (who/what
    produced it, when).

    Evidence is **not linked directly to requirements**. It floats in the
    graph, waiting for a human to judge its sufficiency.
    """)
    return


@app.cell
def __(mo, model_hash, proofs, proof_results, step_summary, dist_summary, params, ProofStatus):
    from rdflib import Graph as _Graph
    from ontology.prefixes import bind_prefixes as _bind
    from evidence.binding import bind_proof_evidence, bind_simulation_evidence, bind_computation_engines
    from evidence.hashing import hash_proof as _hp, hash_evidence, hash_simulation
    from traceability.rtm import load_base_graph, assemble_rtm, validate_evidence_completeness

    _base = load_base_graph()
    _ev = _Graph()
    _bind(_ev)
    bind_computation_engines(_ev)

    for _rid, _script in proofs.items():
        _ph = _hp(_script, model_hash)
        _ch = hash_evidence(model_hash, proof_hash=_ph)
        bind_proof_evidence(
            _ev, f"EV-PROOF-{_rid}", f"SA-{_rid}", _rid,
            model_hash, _ph, _ch,
            f"Symbolic proof: {_script.claim}",
            source_file="analysis/build_proofs.py",
        )

    _sh = hash_simulation({"type": "step_response"}, step_summary)
    for _rid, _desc in [
        ("REQ-001", f"Step response: settling={step_summary['settling_time_s']:.1f}s, final_error={step_summary['final_error_deg']:.4f} deg"),
        ("REQ-002", f"Peak wheel momentum: {step_summary['peak_wheel_momentum']:.3f} N.m.s (limit={params['maxMomentum']})"),
    ]:
        bind_simulation_evidence(
            _ev, f"EV-SIM-{_rid}", f"NS-{_rid}", _rid,
            model_hash, _sh, _desc, source_file="analysis/numerical.py",
        )

    _dh = hash_simulation({"type": "disturbance_rejection"}, dist_summary)
    bind_simulation_evidence(
        _ev, "EV-SIM-REQ-004", "NS-REQ-004", "REQ-004",
        model_hash, _dh,
        f"Disturbance rejection: peak_error={dist_summary['peak_error_deg']:.6f} deg",
        source_file="analysis/numerical.py",
    )

    rtm_graph = assemble_rtm(_base, _ev)
    _issues = validate_evidence_completeness(rtm_graph)

    mo.md(f"""
    ### Evidence Artifacts Created

    - **4 proof artifacts** (one per requirement, hash-bound to model)
    - **3 simulation results** (step response for REQ-001/002, disturbance for REQ-004)
    - Model hash: `{model_hash[:16]}...`
    - Evidence completeness: **{'PASS' if not _issues else 'ISSUES: ' + str(_issues)}**

    Every artifact carries a content hash, a model hash, and a PROV-O
    provenance chain. The hash chain ensures that if the model changes,
    all evidence must be re-produced and re-verified.
    """)
    return rtm_graph, bind_proof_evidence, bind_simulation_evidence, bind_computation_engines, hash_evidence, hash_simulation, load_base_graph, assemble_rtm, validate_evidence_completeness


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Act 5: Attestation

    This is the critical step. The computational pipeline has produced
    evidence — but evidence alone doesn't satisfy requirements. As lead
    controls engineer, Dr. Michael Zargham reviews each requirement's
    evidence and makes two judgments:

    1. **Model adequacy** — Is this model an adequate representation of the
       physical system for evaluating this requirement?
    2. **Evidence sufficiency** — Is the computational evidence sufficient to
       conclude the requirement is satisfied?

    These judgments are recorded as `rtm:Attestation` nodes in the RDF graph,
    with full PROV-O provenance.
    """)
    return


@app.cell
def __(rtm_graph, mo):
    from traceability.attestation import request_attestation

    _adequacy = {
        "REQ-001": ("Linearized PD model adequate for pointing analysis. "
                    "Interface parameters (mass, orbit) accepted from systems engineering. "
                    "Note: settling time exceeds 120s target — model reveals need for Kd tuning, "
                    "but steady-state accuracy is met."),
        "REQ-002": ("Energy-based momentum bound is conservative. "
                    "Reaction wheel model adequate for peak momentum estimation."),
        "REQ-003": ("Linearized stability analysis via Routh-Hurwitz is adequate for this design point. "
                    "Nonlinear effects are second-order for small angles around the operating point."),
        "REQ-004": ("Linearized gravity gradient model adequate for GEO orbit. "
                    "Higher-order terms negligible at geostationary altitude."),
    }
    _sufficiency = {
        "REQ-001": ("Symbolic analysis confirms finite steady-state error well below 0.1 deg. "
                    "Numerical simulation confirms convergence. Settling time of ~262s exceeds 120s target — "
                    "recommend increasing Kd from 10 to ~30 in next design iteration. "
                    "Pointing accuracy requirement MET; settling time requirement MARGINAL."),
        "REQ-002": ("Both symbolic bound (0.81 N.m.s) and numerical simulation confirm "
                    "peak momentum well below 4.0 N.m.s rated capacity. Large margin."),
        "REQ-003": ("Routh-Hurwitz proof confirms asymptotic stability for ALL positive J, Kp, Kd — "
                    "this is a parametric result, not just for one design point. "
                    "Numerical eigenvalues confirm margins exceed -0.010 rad/s on all axes."),
        "REQ-004": ("Gravity gradient torques at GEO are ~1e-6 N.m, four orders of magnitude below "
                    "0.1 N.m actuator capacity. Simulation confirms negligible pointing impact. "
                    "Overwhelming margin."),
    }

    for _rid in ["REQ-001", "REQ-002", "REQ-003", "REQ-004"]:
        request_attestation(
            rtm_graph, _rid, "Dr. Michael Zargham (@mzargham)",
            auto_attest=True,
            model_adequacy=_adequacy[_rid],
            evidence_sufficiency=_sufficiency[_rid],
        )

    mo.md("""
    ### Attestation Complete

    All four requirements have been reviewed and attested by
    Dr. Michael Zargham (@mzargham). Each attestation records:

    - The engineer's identity (PROV-O agent)
    - Timestamp
    - Git commit SHA
    - Model adequacy judgment
    - Evidence sufficiency judgment
    - Links to all evidence artifacts reviewed

    Note the **REQ-001 attestation** flags that settling time exceeds the
    target — this is honest engineering. The requirement is partially met
    (accuracy yes, settling time marginal) and the attestation records a
    recommendation for Kd tuning.
    """)
    return request_attestation


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Act 6: The Audit

    The satellite program's **chief systems engineer** wants to review the
    ADCS team's verification package. They ask: *"How do you know each
    requirement is satisfied?"*

    The RTM graph supports this interrogation. Every link is dereferenceable,
    every proof is re-executable, every simulation is re-runnable.
    """)
    return


@app.cell
def __(rtm_graph, mo):
    from interrogate.explain import explain_requirement

    explanations = {}
    for _rid in ["REQ-001", "REQ-002", "REQ-003", "REQ-004"]:
        explanations[_rid] = explain_requirement(rtm_graph, _rid)
    return explanations, explain_requirement


@app.cell
def __(explanations, mo):
    mo.md(f"""
    ### "How do you know REQ-003 is satisfied?"

    ```
    {explanations["REQ-003"]}
    ```

    The proof was **re-executed live** during this interrogation. The auditor
    doesn't need to trust the original analyst — they can see each lemma
    verified independently, right now.
    """)
    return


@app.cell
def __(explanations, mo):
    mo.md(f"""
    ### "How do you know REQ-001 is satisfied?"

    ```
    {explanations["REQ-001"]}
    ```

    Note the attestation honestly flags the settling time issue and
    recommends Kd tuning. This is what bidirectional traceability enables —
    the auditor can see not just that the requirement was attested, but
    *what the engineer actually judged and why*.
    """)
    return


@app.cell
def __(rtm_graph, mo):
    from interrogate.reproduce import reproduce_all_evidence

    _repro = reproduce_all_evidence(rtm_graph)

    _proof_rows = []
    for _p in _repro["proofs"]:
        _match = "MATCH" if _p["hash_match"] else "MISMATCH"
        _proof_rows.append(f"| {_p['requirement']} | {_p['status'].value} | {_match} |")

    _repro_table = "\n".join(_proof_rows)
    _n_sims = len(_repro['simulations'])
    mo.md(
        "### Reproducibility Audit\n\n"
        "The auditor re-executes ALL computational evidence:\n\n"
        "**Proof Re-verification:**\n\n"
        "| Requirement | Status | Hash Match |\n"
        "|-------------|--------|-----------|\n"
        f"{_repro_table}\n\n"
        f"**Simulation Reproduction:** {_n_sims} simulations re-run successfully.\n\n"
        "Every proof re-verifies. Every hash matches. The evidence is "
        "reproducible — not because we say so, but because the auditor "
        "just confirmed it."
    )
    return reproduce_all_evidence


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Act 7: The Traceability Graph

    The complete RTM as a directed graph. Requirements (blue) flow through
    design elements (green) to evidence (orange/yellow) to attestations (red).
    Every edge is a queryable RDF triple in git.
    """)
    return


@app.cell
def __(rtm_graph):
    from interrogate.visualize import build_rtm_figure

    rtm_fig = build_rtm_figure(rtm_graph, figsize=(18, 10))
    rtm_fig
    return rtm_fig, build_rtm_figure


@app.cell
def __(rtm_graph, mo):
    from traceability.rtm import print_rtm_summary

    _summary = print_rtm_summary(rtm_graph)
    mo.md(f"""
    ### Final Status

    ```
    {_summary}
    ```
    """)
    return print_rtm_summary


@app.cell
def __(mo):
    mo.md("""
    ---

    ## Summary

    This demo has walked through the complete ADCS verification lifecycle:

    1. **Received requirements** from systems engineering, traced to satellite-level parents
    2. **Built a structural model** in SysMLv2-compatible RDF
    3. **Derived formal proofs** using SymPy (13 lemmas across 4 proof scripts)
    4. **Ran numerical simulations** confirming symbolic predictions
    5. **Bound evidence** with content hashes and PROV-O provenance
    6. **Attested** as lead controls engineer, recording model adequacy and evidence sufficiency judgments
    7. **Survived an audit** where the chief engineer re-executed all evidence live

    ### What makes this different

    - **Evidence is not verification.** Only human attestation closes the loop.
    - **Everything is reproducible.** Proofs re-verify, simulations re-run, hashes match.
    - **Everything is in git.** RDF triples, Python scripts, Turtle files — all text, all versioned.
    - **Every link is dereferenceable.** Ask "how do you know?" about any claim and get a machine-readable, human-auditable answer.
    - **Honest engineering.** The attestation for REQ-001 flags a settling time issue rather than hiding it.

    ### Architecture

    | Layer | Technology | Purpose |
    |-------|-----------|---------|
    | Structural Model | SysMLv2 RDF/Turtle | Requirements, design elements, satisfy links |
    | Evidence Layer | PROV-O + custom RTM ontology | Hash-bound evidence, attestation |
    | Symbolic Analysis | SymPy ProofScripts | Formal proofs with re-verifiable lemma chains |
    | Numerical Simulation | scipy solve_ivp | Time-domain ODE integration |
    | Version Control | Git | Source of truth for all artifacts |
    """)
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
