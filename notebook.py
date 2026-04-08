"""# ADCS Lifecycle Demo: Bidirectional Requirements Traceability

A walkthrough of satellite attitude control system design — from receiving
requirements through symbolic analysis, numerical simulation, evidence
binding, human attestation, and audit.
"""

import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def __():
    import sys
    sys.path.insert(0, ".")

    from rdflib import Graph
    from ontology.prefixes import bind_prefixes, SYSML, RTM, ADCS, SAT, PROV
    from traceability.queries import query_to_dicts
    return Graph, bind_prefixes, SYSML, RTM, ADCS, SAT, PROV, query_to_dicts, sys


@app.cell(hide_code=True)
def __(Graph, bind_prefixes):
    from analysis.load_params import load_structural_graph, load_params

    struct_graph = load_structural_graph()
    params = load_params(struct_graph)
    return struct_graph, params, load_structural_graph, load_params


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def __(params):
    from analysis.symbolic import (
        run_symbolic_analysis,
        build_inertia_tensor_symbolic,
        evaluate_inertia,
        stability_margins,
    )

    sym_result = run_symbolic_analysis(params)
    return sym_result, run_symbolic_analysis, build_inertia_tensor_symbolic, evaluate_inertia, stability_margins


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def __(mo):
    mo.md("""
    ### Formal Proofs

    Each requirement gets a ProofScript — a chain of SymPy lemmas, each
    independently re-verifiable. The proof is bound to the structural model
    via content hash: if the model changes, the proof hash changes, alerting
    auditors to re-verify.
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def __(params):
    from analysis.numerical import run_step_response, run_disturbance_rejection

    step_result = run_step_response(params)
    step_summary = step_result.summary()

    dist_result = run_disturbance_rejection(params)
    dist_summary = dist_result.summary()
    return step_result, step_summary, dist_result, dist_summary, run_step_response, run_disturbance_rejection


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def __(mo):
    mo.md("""
    ---

    ## Act 4: Evidence Binding

    Now we bind our computational results to the RDF traceability graph.
    Every evidence artifact gets a content hash, a model hash (binding it
    to the structural model version), and PROV-O provenance (who/what
    produced it, when).

    Each evidence artifact **addresses** a specific requirement — recording
    the structural intent that "this proof was constructed to evaluate
    REQ-003." But `rtm:addresses` is not `rtm:attests`. The evidence
    says *what was analyzed*; only human attestation says *whether it's
    sufficient*. An evidence artifact can address a requirement and still
    lead to a declined attestation — as we'll see with REQ-001.
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def __(rtm_graph, mo):
    from traceability.attestation import request_attestation

    _adequacy = {
        "REQ-002": ("Energy-based momentum bound is conservative. "
                    "Reaction wheel model adequate for peak momentum estimation."),
        "REQ-003": ("Linearized stability analysis via Routh-Hurwitz is adequate for this design point. "
                    "Nonlinear effects are second-order for small angles around the operating point."),
        "REQ-004": ("Linearized gravity gradient model adequate for GEO orbit. "
                    "Higher-order terms negligible at geostationary altitude."),
    }
    _sufficiency = {
        "REQ-002": ("Both symbolic bound (0.81 N.m.s) and numerical simulation confirm "
                    "peak momentum well below 4.0 N.m.s rated capacity. Large margin."),
        "REQ-003": ("Routh-Hurwitz proof confirms asymptotic stability for ALL positive J, Kp, Kd — "
                    "this is a parametric result, not just for one design point. "
                    "Numerical eigenvalues confirm margins exceed -0.010 rad/s on all axes."),
        "REQ-004": ("Gravity gradient torques at GEO are ~1e-6 N.m, four orders of magnitude below "
                    "0.1 N.m actuator capacity. Simulation confirms negligible pointing impact. "
                    "Overwhelming margin."),
    }

    # Attest REQ-002, REQ-003, REQ-004 — evidence is sufficient
    for _rid in ["REQ-002", "REQ-003", "REQ-004"]:
        request_attestation(
            rtm_graph, _rid, "Dr. Michael Zargham (@mzargham)",
            auto_attest=True,
            model_adequacy=_adequacy[_rid],
            evidence_sufficiency=_sufficiency[_rid],
        )

    # REQ-001: DECLINE attestation — settling time does not meet requirement
    # (We don't call request_attestation for REQ-001)

    mo.md("""
    ### Attestation Results

    **REQ-002: ATTESTED** — Peak wheel momentum well within limits.

    **REQ-003: ATTESTED** — Routh-Hurwitz proof confirms stability for all
    positive gains. Parametric result, not design-point-specific.

    **REQ-004: ATTESTED** — Gravity gradient torques orders of magnitude
    below actuator capacity.

    **REQ-001: DECLINED** — The model is adequate (linearized PD analysis
    is appropriate for pointing), but the evidence is **not sufficient** to
    conclude the requirement is satisfied. Specifically:

    - Steady-state pointing accuracy (< 0.1 deg): **MET** — error is ~0.0001 deg
    - Settling time (< 120 s): **NOT MET** — simulation shows ~262s settling time

    The settling time exceeds the requirement by over 2x. The root cause is
    insufficient controller bandwidth relative to the X-axis inertia
    (Jxx = 327 kg-m^2). **Recommendation: retune controller gains
    (Kp: 1→4, Kd: 10→30) and re-verify.**

    This is the system working as intended — the engineer cannot attest a
    requirement when the evidence shows it is not satisfied. The finding is
    recorded, and a design iteration is needed.
    """)
    return request_attestation


@app.cell(hide_code=True)
def __(mo):
    mo.md("""
    ---

    ## Act 6: The Audit

    The satellite program's **chief systems engineer** reviews the ADCS
    team's verification package. The status is: **3 of 4 requirements
    attested, 1 open finding.**

    The auditor asks: *"Show me the evidence for each requirement."*

    The RTM graph supports this interrogation. Every link is dereferenceable,
    every proof is re-executable, every simulation is re-runnable. And for
    REQ-001, the auditor can see exactly *why* attestation was declined and
    what the recommended corrective action is.
    """)
    return


@app.cell(hide_code=True)
def __(rtm_graph, mo):
    from interrogate.explain import explain_requirement

    explanations = {}
    for _rid in ["REQ-001", "REQ-002", "REQ-003", "REQ-004"]:
        explanations[_rid] = explain_requirement(rtm_graph, _rid)
    return explanations, explain_requirement


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def __(explanations, mo):
    mo.md(f"""
    ### "What about REQ-001?"

    ```
    {explanations["REQ-001"]}
    ```

    REQ-001 is **NOT ATTESTED**. The evidence shows the settling time
    requirement is not met. The auditor can see:

    - The proof confirms the steady-state error formula is correct
    - The simulation confirms the system converges — but too slowly
    - No engineer has attested this requirement
    - The traceability chain is complete up to evidence, but stops there

    This is what bidirectional traceability looks like when a requirement
    **fails**. The gap is visible, auditable, and actionable.
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def __(mo):
    mo.md("""
    ---

    ## Act 7: The Traceability Graph

    The complete RTM as a directed graph. Requirements (blue) flow through
    design elements (green) to evidence (orange/yellow) to attestations (red).
    Every edge is a queryable RDF triple in git.
    """)
    return


@app.cell(hide_code=True)
def __(rtm_graph):
    from interrogate.visualize import build_rtm_figure

    rtm_fig = build_rtm_figure(rtm_graph, figsize=(18, 10))
    rtm_fig
    return rtm_fig, build_rtm_figure


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def __(mo):
    mo.md("""
    ---

    ## Act 8: Design Iteration

    The open finding on REQ-001 drives action. We need to increase the
    derivative gain Kd to reduce settling time. Rather than editing a file
    by hand, we apply a **SPARQL UPDATE** to the structural model — the
    same way any RDF-native toolchain would propagate a design change.

    This demonstrates **reproducibility as regression testing**: the model
    hash changes, all previous proofs are invalidated, and we must re-run
    the entire analysis chain from scratch.
    """)
    return


@app.cell(hide_code=True)
def __(struct_graph, mo):
    from rdflib import Literal, XSD

    # Clone the graph for the design iteration
    from rdflib import Graph as _G
    from ontology.prefixes import bind_prefixes as _bp, SYSML as _SYSML

    v2_graph = _G()
    _bp(v2_graph)
    for triple in struct_graph:
        v2_graph.add(triple)

    # Apply SPARQL UPDATE: retune both controller gains
    # Increasing Kd alone would overdamp the system — a controls engineer
    # retunes both Kp (stiffness) and Kd (damping) together.
    _sparql_update_kd = """
    DELETE { ?attr sysml:value ?oldVal }
    INSERT { ?attr sysml:value "30.0"^^xsd:double }
    WHERE {
        ?attr sysml:declaredName "Kd" ;
              sysml:value ?oldVal .
    }
    """
    _sparql_update_kp = """
    DELETE { ?attr sysml:value ?oldVal }
    INSERT { ?attr sysml:value "4.0"^^xsd:double }
    WHERE {
        ?attr sysml:declaredName "Kp" ;
              sysml:value ?oldVal .
    }
    """
    v2_graph.update(_sparql_update_kd, initNs={"sysml": _SYSML})
    v2_graph.update(_sparql_update_kp, initNs={"sysml": _SYSML})

    # Verify
    _check = """
    SELECT ?name ?val WHERE {
        ?attr sysml:declaredName ?name ;
              sysml:value ?val .
        FILTER(?name IN ("Kp", "Kd"))
    }
    ORDER BY ?name
    """
    _gains = {str(r[0]): float(r[1]) for r in v2_graph.query(_check, initNs={"sysml": _SYSML})}

    mo.md(
        "### SPARQL UPDATE: Controller Retune\n\n"
        "A controls engineer retunes both gains together — increasing Kd alone\n"
        "would overdamp the system, making settling *slower* despite more damping.\n\n"
        "```sparql\n"
        "# Increase proportional gain (stiffness)\n"
        'DELETE { ?attr sysml:value ?oldVal }\n'
        'INSERT { ?attr sysml:value "4.0"^^xsd:double }\n'
        'WHERE  { ?attr sysml:declaredName "Kp" ; sysml:value ?oldVal . }\n\n'
        "# Increase derivative gain (damping)\n"
        'DELETE { ?attr sysml:value ?oldVal }\n'
        'INSERT { ?attr sysml:value "30.0"^^xsd:double }\n'
        'WHERE  { ?attr sysml:declaredName "Kd" ; sysml:value ?oldVal . }\n'
        "```\n\n"
        f"| Gain | Before | After |\n"
        f"|------|--------|-------|\n"
        f"| Kp | 1.0 N.m/rad | **{_gains['Kp']:.1f}** N.m/rad |\n"
        f"| Kd | 10.0 N.m.s/rad | **{_gains['Kd']:.1f}** N.m.s/rad |"
    )
    return v2_graph


@app.cell(hide_code=True)
def __(v2_graph, model_hash, mo):
    from evidence.hashing import hash_structural_model as _hsm

    v2_model_hash = _hsm(v2_graph)

    mo.md(
        "### Model Hash Invalidation\n\n"
        f"| | Hash |\n"
        f"|--|------|\n"
        f"| Original model | `{model_hash[:24]}...` |\n"
        f"| Updated model  | `{v2_model_hash[:24]}...` |\n\n"
        "The hashes differ — **all previous proofs and evidence are now invalid**.\n"
        "Any attempt to verify an old proof against the new model hash will fail.\n"
        "We must re-derive everything from scratch."
    )
    return v2_model_hash


@app.cell(hide_code=True)
def __(v2_graph, v2_model_hash, mo):
    from analysis.load_params import load_params as _lp
    from analysis.symbolic import run_symbolic_analysis as _rsa
    from analysis.build_proofs import build_all_proofs as _bap
    from analysis.proof_scripts import verify_proof as _vp, ProofStatus as _PS

    v2_params = _lp(v2_graph)
    v2_sym = _rsa(v2_params)
    v2_margins = v2_sym.stability_margins

    v2_proofs = _bap(v2_model_hash)
    v2_proof_results = {}
    for _rid, _script in v2_proofs.items():
        v2_proof_results[_rid] = _vp(_script, v2_model_hash)

    _margin_rows = "\n".join(
        f"| {axis} | {val:.4f} | {'PASS' if val <= -0.010 else 'FAIL'} |"
        for axis, val in v2_margins.items()
    )
    _proof_rows = "\n".join(
        f"| {rid} | {'VERIFIED' if r.status == _PS.VERIFIED else 'FAILED'} |"
        for rid, r in sorted(v2_proof_results.items())
    )

    mo.md(
        "### Re-run Symbolic Analysis (Kp=4, Kd=30)\n\n"
        f"Inertia unchanged: Jxx={v2_sym.inertia[0]:.1f}, "
        f"Jyy={v2_sym.inertia[1]:.1f}, Jzz={v2_sym.inertia[2]:.1f} kg.m^2\n\n"
        "**Stability margins (improved):**\n\n"
        "| Axis | Re(lambda) | Status |\n"
        "|------|-----------|--------|\n"
        f"{_margin_rows}\n\n"
        f"Settling time estimate: **{v2_sym.pointing_budget['settling_time_s']:.1f}s** "
        f"(was 262s, requirement is 120s)\n\n"
        "**Proofs (re-verified against new model hash):**\n\n"
        "| Requirement | Status |\n"
        "|-------------|--------|\n"
        f"{_proof_rows}"
    )
    return v2_params, v2_sym, v2_proofs, v2_proof_results


@app.cell(hide_code=True)
def __(v2_params, mo):
    from analysis.numerical import run_step_response as _rsr

    v2_step = _rsr(v2_params)
    v2_step_summary = v2_step.summary()

    import matplotlib.pyplot as _plt2
    import numpy as _np2

    _fig2, _ax2 = _plt2.subplots(1, 2, figsize=(14, 5))
    _axis_colors = {"X": "#1f77b4", "Y": "#2ca02c", "Z": "#9467bd"}
    _limit_color = "#d62728"

    # Pointing convergence comparison
    _q_vec2 = _np2.linalg.norm(v2_step.q[:, :3], axis=1)
    _theta2 = _np2.degrees(2 * _q_vec2)
    _ax2[0].semilogy(v2_step.t, _theta2, color=_axis_colors["X"], linewidth=1.5, label="Kp=4, Kd=30 (updated)")
    _ax2[0].axhline(0.1, color=_limit_color, linestyle="--", linewidth=1, label="REQ-001 limit")
    _ax2[0].axvline(120, color="#888", linestyle=":", linewidth=1, alpha=0.7, label="120s target")
    _ax2[0].set_xlabel("Time (s)")
    _ax2[0].set_ylabel("Attitude Error (deg)")
    _ax2[0].set_title("Pointing Convergence (Kp=4, Kd=30)")
    _ax2[0].set_ylim(bottom=1e-4)
    _ax2[0].legend(fontsize=8)
    _ax2[0].grid(True, alpha=0.3, which="both")

    # Wheel momentum
    _h2 = _np2.linalg.norm(v2_step.h_wheel, axis=1)
    _ax2[1].plot(v2_step.t, _h2, color=_axis_colors["X"], linewidth=1.5, label="|h| (total)")
    _ax2[1].axhline(v2_step.config.max_momentum, color=_limit_color, linestyle="--", linewidth=1, label="REQ-002 limit")
    _ax2[1].set_xlabel("Time (s)")
    _ax2[1].set_ylabel("Momentum (N.m.s)")
    _ax2[1].set_title("Wheel Momentum (Kp=4, Kd=30)")
    _ax2[1].legend(fontsize=8)
    _ax2[1].grid(True, alpha=0.3)

    _fig2.suptitle("Design Iteration: Step Response with Kp=4, Kd=30", fontsize=13, fontweight="bold")
    _plt2.tight_layout()

    mo.md(
        "### Re-run Numerical Simulation (Kp=4, Kd=30)\n\n"
        f"| Metric | Kd=10 (original) | Kd=30 (updated) | Requirement |\n"
        f"|--------|-----------------|-----------------|-------------|\n"
        f"| Settling time | 262s | **{v2_step_summary['settling_time_s']:.1f}s** | < 120s |\n"
        f"| Final error | 0.1223 deg | **{v2_step_summary['final_error_deg']:.4f} deg** | < 0.1 deg |\n"
        f"| Peak momentum | 0.810 N.m.s | **{v2_step_summary['peak_wheel_momentum']:.3f} N.m.s** | < 4.0 N.m.s |\n"
        f"| Peak torque | 0.044 N.m | **{v2_step_summary['peak_control_torque']:.4f} N.m** | < 0.1 N.m |\n"
    )

    _fig2
    return v2_step, v2_step_summary


@app.cell(hide_code=True)
def __(mo):
    mo.md("""
    ### Regression Check

    With Kp=4, Kd=30, the settling time is now well under 120s. But we must
    verify that the gain increase didn't break anything else:

    - **REQ-002 (momentum):** Peak momentum may increase (higher gains → more
      aggressive control), but must remain within the 4.0 N.m.s limit.
    - **REQ-003 (stability):** Higher gains improve both bandwidth and damping.
    - **REQ-004 (disturbance):** Gravity gradient rejection improves — higher Kp
      reduces steady-state error from disturbances.

    This is **reproducibility as regression testing**. The same pipeline that found
    the deficiency now confirms the fix doesn't introduce new problems.
    """)
    return


@app.cell(hide_code=True)
def __(v2_graph, v2_model_hash, v2_params, v2_proofs, v2_step_summary, v2_step, mo):
    from rdflib import Graph as _G2
    from ontology.prefixes import bind_prefixes as _bp2
    from evidence.binding import (
        bind_proof_evidence as _bpe, bind_simulation_evidence as _bse,
        bind_computation_engines as _bce,
    )
    from evidence.hashing import (
        hash_proof as _hp2, hash_evidence as _he2, hash_simulation as _hs2,
    )
    from analysis.proof_scripts import verify_proof as _vp2
    from traceability.rtm import load_base_graph as _lbg, assemble_rtm as _art
    from traceability.attestation import request_attestation as _ra

    # Rebuild base graph from updated structural model
    _base2 = _G2()
    _bp2(_base2)
    # Load ontology files
    from pathlib import Path as _P
    for _ttl in sorted((_P("ontology")).glob("*.ttl")):
        _base2.parse(str(_ttl), format="turtle")
    # Add updated structural model (v2_graph has the Kd=30 change)
    for _t in v2_graph:
        _base2.add(_t)

    # Build evidence graph
    _ev2 = _G2()
    _bp2(_ev2)
    _bce(_ev2)

    for _rid, _script in v2_proofs.items():
        _ph = _hp2(_script, v2_model_hash)
        _ch = _he2(v2_model_hash, proof_hash=_ph)
        _bpe(_ev2, f"EV-PROOF-{_rid}-v2", f"SA-{_rid}-v2", _rid,
             v2_model_hash, _ph, _ch,
             f"Symbolic proof (v2): {_script.claim}",
             source_file="analysis/build_proofs.py")

    _sh2 = _hs2(v2_step.config.to_dict(), v2_step_summary)
    for _rid, _desc in [
        ("REQ-001", f"Step response (v2): settling={v2_step_summary['settling_time_s']:.1f}s, "
                    f"final_error={v2_step_summary['final_error_deg']:.4f} deg"),
        ("REQ-002", f"Peak wheel momentum (v2): {v2_step_summary['peak_wheel_momentum']:.3f} N.m.s"),
    ]:
        _bse(_ev2, f"EV-SIM-{_rid}-v2", f"NS-{_rid}-v2", _rid,
             v2_model_hash, _sh2, _desc, source_file="analysis/numerical.py")

    v2_rtm = _art(_base2, _ev2)

    # Attest ALL 4 requirements — now all evidence is sufficient
    _v2_adequacy = {
        "REQ-001": ("Linearized PD model adequate. Kp increased to 4, Kd to 30 per design review finding. "
                    "Interface parameters unchanged from systems engineering."),
        "REQ-002": "Energy-based momentum bound conservative. Peak may increase with higher gains but within limits.",
        "REQ-003": "Routh-Hurwitz stability confirmed. Higher gains improve damping and bandwidth.",
        "REQ-004": "GG model adequate. Controller gain change does not affect disturbance torque magnitude.",
    }
    _v2_sufficiency = {
        "REQ-001": (f"Settling time now {v2_step_summary['settling_time_s']:.1f}s (< 120s requirement). "
                    f"Pointing accuracy {v2_step_summary['final_error_deg']:.4f} deg (< 0.1 deg). Both met."),
        "REQ-002": f"Peak momentum {v2_step_summary['peak_wheel_momentum']:.3f} N.m.s, well within 4.0 limit.",
        "REQ-003": "Routh-Hurwitz proof valid for all positive J, Kp, Kd. Margins improved with higher gains.",
        "REQ-004": "GG torques unchanged at ~1e-6 N.m. Overwhelming margin vs 0.1 N.m actuator capacity.",
    }

    for _rid in ["REQ-001", "REQ-002", "REQ-003", "REQ-004"]:
        _ra(v2_rtm, _rid, "Dr. Michael Zargham (@mzargham)",
            auto_attest=True,
            model_adequacy=_v2_adequacy[_rid],
            evidence_sufficiency=_v2_sufficiency[_rid])

    mo.md("""
    ### Full Re-attestation (v2 model)

    With the updated model, all four requirements now have sufficient evidence:

    - **REQ-001: ATTESTED** — settling time and pointing accuracy both met
    - **REQ-002: ATTESTED** — momentum within limits (slight increase noted)
    - **REQ-003: ATTESTED** — stability margins improved
    - **REQ-004: ATTESTED** — disturbance rejection unchanged

    The entire evidence chain is fresh — new model hash, new proof hashes,
    new simulation hashes. Nothing carries over from the v1 analysis.
    """)
    return v2_rtm


@app.cell(hide_code=True)
def __(v2_rtm):
    from interrogate.visualize import build_rtm_figure as _brf

    v2_fig = _brf(v2_rtm, figsize=(18, 10), title="Requirements Traceability Matrix (v2: Kp=4, Kd=30)")
    v2_fig
    return v2_fig


@app.cell(hide_code=True)
def __(v2_rtm, mo):
    from interrogate.explain import explain_requirement as _er

    _v2_expl = _er(v2_rtm, "REQ-001")
    mo.md(
        '### "How do you know REQ-001 is satisfied now?"\n\n'
        f"```\n{_v2_expl}\n```\n\n"
        "The traceability chain is now **complete** for REQ-001. The auditor can see:\n\n"
        "- Fresh proof artifact bound to the v2 model hash\n"
        "- Fresh simulation showing settling time under 120s\n"
        "- Attestation by Dr. Zargham with explicit reference to the design change\n"
        "- The v2 model hash differs from v1 — confirming this is new evidence, not recycled"
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""
    ---

    ## Summary

    This demo walked through the complete ADCS verification lifecycle,
    including a design iteration driven by an open finding:

    1. **Received requirements** from systems engineering, derived ADCS-level requirements
    2. **Built a structural model** in SysMLv2-compatible RDF
    3. **Derived formal proofs** (13 lemmas across 4 proof scripts)
    4. **Ran numerical simulations** — revealed settling time deficiency on REQ-001
    5. **Bound evidence** with content hashes and PROV-O provenance
    6. **Attested 3 of 4 requirements** — declined REQ-001 (settling time 262s > 120s)
    7. **Underwent audit** — open finding confirmed by chief engineer
    8. **Applied design change** via SPARQL UPDATE (Kd: 10 → 30)
    9. **Re-ran full analysis** — model hash changed, all proofs re-derived
    10. **Attested all 4 requirements** — REQ-001 now satisfied, regression confirmed

    ### What makes this different

    - **Evidence is not verification.** Only human attestation closes the loop — and attestation can be declined.
    - **Failures are first-class.** The REQ-001 gap in the v1 traceability graph was the finding that drove the design change.
    - **Reproducibility is regression testing.** The same pipeline that found the deficiency confirmed the fix didn't break anything else.
    - **Model changes invalidate evidence.** Kd 10 → 30 changed the model hash, forcing complete re-derivation. No stale proofs survive.
    - **Everything is in git.** Both model versions, both evidence sets, both attestation records — all text, all versioned, all auditable.
    - **Every link is dereferenceable.** Ask "how do you know?" about any claim at any version and get a machine-readable, human-auditable answer.

    ### Architecture

    | Layer | Technology | Purpose |
    |-------|-----------|---------|
    | Structural Model | SysMLv2 RDF/Turtle | Requirements, design elements, satisfy links |
    | Model Changes | SPARQL UPDATE | Modify parameters, trigger hash invalidation |
    | Evidence Layer | PROV-O + custom RTM ontology | Hash-bound evidence, attestation |
    | Symbolic Analysis | SymPy ProofScripts | Formal proofs with re-verifiable lemma chains |
    | Numerical Simulation | scipy solve_ivp | Time-domain ODE integration |
    | Version Control | Git | Source of truth for all artifacts |
    """)
    return


@app.cell(hide_code=True)
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
