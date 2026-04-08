# ADCS Lifecycle Demo

Bidirectional requirements traceability for a satellite Attitude Determination and Control System (ADCS), demonstrating the full lifecycle from SysMLv2 structural specification through symbolic analysis, numerical simulation, and human expert attestation.

## Core Principle

**Evidence does not verify requirements; evidence supports a human judgment that requirements are satisfied.**

Models are imperfect representations of physical systems. Symbolic proofs and simulation results are claims true *within the model*. The engineer judges model adequacy and evidence sufficiency. Only human attestation connects evidence to requirement satisfaction.

## Architecture

Three-layer ontology over git-native RDF:

| Layer | Vocabulary | Purpose |
|-------|-----------|---------|
| W3C Standards | PROV-O, Dublin Core | Provenance chains, metadata |
| SysMLv2 | `sysml:` | Structural model, requirements, satisfy relationships |
| RTM | `rtm:` | Evidence, attestation, traceability |

The demo is scoped as the **ADCS controls team** — one disciplinary team within a broader satellite program. Requirements derive from satellite-level requirements; interface parameters (mass, orbit) come from systems engineering.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the full pipeline (auto-attestation)
uv run python -m pipeline.runner --auto

# Run interactively (prompts for attestation judgments)
uv run python -m pipeline.runner

# Run tests
uv run pytest -v
```

## Pipeline Stages

1. **Structural Model** — Load SysMLv2 RDF (satellite hierarchy, ADCS components, 4 requirements with satisfy links)
2. **Symbolic Analysis** — SymPy derives inertia tensors, eigenvalues, stability proofs (Routh-Hurwitz), pointing error bounds
3. **Numerical Simulation** — scipy integrates quaternion + Euler dynamics with PD control and gravity gradient disturbance
4. **Evidence Binding** — Hash-chained evidence artifacts in RDF with PROV-O provenance
5. **RTM Assembly** — Merge structural + evidence graphs, validate completeness
6. **Attestation** — Engineer reviews evidence, judges model adequacy and evidence sufficiency
7. **Reporting** — Export final RTM as Turtle, print status summary
8. **Interrogation** — "How do you know X?" with live proof re-verification

## Interrogation

```python
from pipeline.runner import run_pipeline
from interrogate.explain import explain_requirement

rtm = run_pipeline(auto_attest=True)
print(explain_requirement(rtm, "REQ-003"))
```

Produces a dereferenceable explanation chain:

```
REQ-003: "The closed-loop ADCS shall be asymptotically stable..."
├── Derived from: SAT-REQ-STABILITY (satellite-level)
├── Allocated to: PDController
├── Evidence (1 artifacts):
│   └── [ProofArtifact]
│   │   Re-verification: VERIFIED (re-executed just now)
│   │     characteristic_polynomial_form: ✓
│   │     routh_row0_positive: ✓
│   │     routh_row1_positive: ✓
│   │     routh_row2_positive: ✓
└── Attestation:
    Attested by: Dr. Michael Zargham (@mzargham)
    Model adequacy: "Linearized stability analysis via Routh-Hurwitz
                     is adequate..."
    Evidence sufficiency: "Routh-Hurwitz proof confirms asymptotic
                          stability for all positive J, Kp, Kd..."
```

## Requirements

| ID | Requirement | Evidence |
|----|------------|---------|
| REQ-001 | Pointing accuracy < 0.1 deg within 120s | Steady-state error bound + step response simulation |
| REQ-002 | Wheel momentum < 4.0 N.m.s | Peak momentum bound + simulation |
| REQ-003 | Closed-loop stability: Re(lambda) <= -0.010 | Routh-Hurwitz formal proof |
| REQ-004 | Gravity gradient rejection at GEO | Disturbance torque analysis + simulation |

## License

Apache 2.0 — see [LICENSE](LICENSE).
