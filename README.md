# ADCS Lifecycle Demo

**Live demo:** <https://dynamicalsystemsgroup.github.io/ADCS-lifecycle-demo/>

Bidirectional requirements traceability for a satellite Attitude Determination
and Control System (ADCS), demonstrating the full lifecycle from SysMLv2
structural specification through symbolic analysis, numerical simulation, and
human expert attestation. The RTM is held as an `rdflib.Dataset` of named
graphs, validated by a SHACL closure-rule suite, audited forward and backward
independently, and exportable to disk, to a Flexo MMS instance, or to bare
Apache Jena Fuseki.

## Core Principle

**Evidence does not verify requirements; evidence supports a human judgment
that requirements are satisfied.**

Models are imperfect representations of physical systems. Symbolic proofs and
simulation results are claims true *within the model*. The engineer judges
**model adequacy** and **evidence sufficiency**. Only human attestation
connects evidence to requirement satisfaction.

This split is not novel — it is the canonical Hawkins–Habli Assurance Claim
Point categorization. In this demo, **adequacy** is a `gsn:Assumption` and
**sufficiency** is a `gsn:Justification`, both linked to the attestation via
`gsn:inContextOf`. `rtm:` adds no novel epistemic vocabulary; it is a thin
integration layer over established standards.

## Architecture

Integration ontology assembled from W3C and OMG standards plus the OMG SysMLv2
OWL rendering:

| Layer        | Vocabulary                           | Role                                                              |
| ------------ | ------------------------------------ | ----------------------------------------------------------------- |
| W3C / IETF   | `prov:`, `dcterms:`, `earl:`, `sh:`  | Provenance + assertion + outcome + SHACL closure                  |
| OMG / SysML  | `sysml:` ↔ `omg-sysml:`              | Structural model + requirements (aliased to the OMG SysMLv2 OWL rendering via owl:equivalentClass) |
| Community    | `gsn:`, `p-plan:`                    | Assurance argument structure + declarative process model         |
| Tool interop | `oslc_rm:`, `oslc_qm:`               | Aliases for DOORS Next / Jama / RQM                              |
| Local glue   | `rtm:`                               | Convenience subclasses + content-addressing properties only      |

See [`ontology/rtm-edit.ttl`](ontology/rtm-edit.ttl) for the source and
[`ontology/assembly_manifest.json`](ontology/assembly_manifest.json) for the
build-time manifest.

## Named-Graph Quadstore Layout

The runtime holds the RTM as an `rdflib.Dataset` with one named graph per
content layer — sized to match how Flexo MMS partitions projects/branches:

| Named graph             | Contents                                                |
| ----------------------- | ------------------------------------------------------- |
| `<rtm:ontology>`        | TBox + shapes + individuals (from `ontology/*.ttl`)     |
| `<rtm:plan>`            | P-PLAN process model (`pipeline/plan.ttl`)              |
| `<adcs:structural>`     | SysMLv2 instance data (satellite + parameters)         |
| `<adcs:context>`        | Stable `gsn:Context` / `gsn:Assumption` individuals    |
| `<adcs:evidence>`       | Generated `rtm:ProofArtifact` / `rtm:SimulationResult` |
| `<adcs:attestations>`   | `rtm:Attestation` events + adequacy / sufficiency nodes |
| `<adcs:plan-execution>` | `p-plan:Activity` instances (one per stage)            |
| `<adcs:audit>`          | Audit report summary triples (forward/backward results) |

SPARQL queries use `Dataset(default_union=True)` so existing queries match
across the union without `GRAPH` clauses.

## Quick Start

```bash
uv sync
uv run python -m pipeline.runner --auto       # scripted attestation
uv run python -m pipeline.runner               # interactive attestation
uv run pytest -v                                # 166 tests
```

## Pipeline Stages

```
[Stage 0]   Ontology Assembly         — narrate the integration; load TBox into <rtm:ontology>
[Stage 1]   Structural Model          — load SysMLv2 instance data into <adcs:structural>
[Stage 2]   Symbolic Analysis         — SymPy: inertia, eigenvalues, stability proofs
[Stage 3]   Numerical Simulation      — scipy: step response + disturbance rejection
[Stage 4]   Evidence Binding          — emit hashed evidence into <adcs:evidence>
[Stage 5]   RTM Assembled             — validate evidence completeness
[Stage 6]   Human Attestation         — emit attestations into <adcs:attestations>
[Stage 6.5] Validate Closure-Rule Suite — 9 SHACL shapes + runtime re-verification
[Stage 7a]  Audit Trace               — forward / backward / bidirectional + coverage matrix
[Stage 7]   Generate Reports          — persist via backend, write output/audit.{md,csv}
[Stage 8]   Interrogate               — explain / reproduce / visualize ready
```

### Stage 0 banner (first thing you see)

```
[Stage 0/8] Ontology Assembly
─────────────────────────────────────────────────────────────────
  Loading assembled rtm.ttl (built 2026-05-14T01:26:27Z from rtm-edit.ttl)
  Imports resolved:
    OK  EARL         162 triples,  5 terms referenced
    OK  OSLC QM       90 triples,  1 terms referenced
    OK  OSLC RM       74 triples,  1 terms referenced
    OK  OntoGSN     3784 triples,  3 terms referenced
    OK  P-PLAN       154 triples,  0 terms referenced
    OK  PROV-O      1146 triples,  7 terms referenced
  SysMLv2 equivalence axioms: 9
  Local rtm: integration glue: 13 subclass + 7 subproperty axioms
  Validation: Python build (run `make ontology-robot` for ELK + report)
  Loaded into <rtm:ontology>: 317 triples
  Closure-rule suite registered: 13 SHACL shapes
─────────────────────────────────────────────────────────────────
```

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
├── Derived from: SAT-REQ-STABILITY
├── Allocated to: PDController
├── Evidence (1 artifacts):
│   └── [ProofArtifact]
│       Re-verification: VERIFIED (re-executed just now)
│         characteristic_polynomial_form: ✓
│         routh_row0_positive: ✓
│         routh_row1_positive: ✓
│         routh_row2_positive: ✓
└── Attestation:
    Outcome: earl:passed (mode: earl:semiAuto)
    Attested by: Dr. Michael Zargham (@mzargham)
    Model adequacy: "Linearized stability analysis via Routh-Hurwitz..."
    Evidence sufficiency: "Routh-Hurwitz proof confirms asymptotic..."
```

## Requirements

| ID      | Requirement                                       | Outcome (default run)   |
| ------- | ------------------------------------------------- | ----------------------- |
| REQ-001 | Pointing accuracy < 0.1 deg within 120s            | `earl:failed` (declined — settling time exceeds spec) |
| REQ-002 | Wheel momentum < 4.0 N.m.s                         | `earl:passed`           |
| REQ-003 | Closed-loop stability: Re(λ) ≤ -0.010              | `earl:passed`           |
| REQ-004 | Gravity gradient rejection at GEO                  | `earl:passed`           |

REQ-001 is intentionally a declined attestation (a well-formed
`rtm:Attestation` with `earl:failed` outcome) so the closure-rule suite has a
realistic case where the audit must distinguish "no attestation" from
"attested-but-not-passing."

## Closure-Rule Suite

Ten machine-checkable invariants. Nine are SHACL shapes in
[`ontology/rtm_shapes.ttl`](ontology/rtm_shapes.ttl); the tenth is a runtime
re-verification check (`interrogate/reproduce`).

1. `AttestationShape` — adequacy Assumption + sufficiency Justification + outcome + qualified association
2. `PlanInstantiationShape` — every `p-plan:Activity` corresponds to a known step
3. `EvidenceShape` — hashes + source + generating activity + addresses a requirement
4. `RequirementShape` — declared name + text
5. `GsnArgumentShape` — Goal → Strategy → Solution chain; non-empty statements
6. `ProvenanceShape` — Activity has Agent; Entity has ≤1 generating activity
7. `OutcomeSemanticsShape` — outcome-specific rationale requirements
8. `ForwardTraceabilityShape` / `BackwardTraceabilityShape` — *independent*; bidirectional is the conjunction
9. `NamedGraphIntegrityShape` — cross-graph references resolve
10. **Re-verification closure** (runtime) — every `rtm:ProofArtifact` re-hashes to its stored `rtm:proofHash`

Run at pipeline time as Stage 6.5; report lands in `<adcs:audit>`.

## Audit (Stage 7a)

Forward and backward traceability checks run *independently* and emit
separate failure lists, so error messages identify which direction broke:

```
[Stage 7a] Auditing forward / backward / bidirectional traceability...
  Forward    PASS  (4 checked, 0 failures)
  Backward   PASS  (4 checked, 0 failures)
  Bidirectional: PASS
  Orphans: none
  Audit report -> output/audit.md, output/audit.csv
```

Three diagnostic shapes Jama-style:

- *forward-fail, backward-pass*: a requirement isn't reached by any evidence
- *backward-fail, forward-pass*: an attestation references evidence that doesn't address the same requirement
- *both-fail*: both message sets reported, each direction labeled

CLI: `uv run python -m traceability.audit --direction {forward|backward|bidirectional|full} --format {csv|md|json}`

## Persistence Backends

The runtime always builds the Dataset locally. The `--backend` flag chooses
where it ultimately lands:

```bash
# (default) write output/rtm.ttl + output/rtm.trig to disk
uv run python -m pipeline.runner --auto --backend=local

# push each named graph to a Flexo MMS instance
uv run python -m pipeline.runner --auto --backend=flexo

# push to a bare Apache Jena Fuseki (no-Flexo fallback)
uv run python -m pipeline.runner --auto --backend=fuseki
```

The Flexo backend targets the shared remote sandbox at
`try-layer1.starforge.app` by default — see [`flexo/README.md`](flexo/README.md)
for the token-based remote workflow and the local Compose-stack alternative.

Audit results are identical across all three backends: the SPARQL queries run
against the local Dataset; the backend only persists. Demonstrated live:

```
=== branches in adcs-demo/lifecycle on starforge ===
attestations  audit  evidence  master  ontology  plan-execution  structural

=== SPARQL triple counts per branch ===
ontology         317       evidence         126
structural       253       attestations      89
plan-execution    52       audit              8
```

## Compute Backends (Phase L)

The Stage 2 / Stage 3 analyses can optionally run inside a Docker container,
emulating a remote analysis server. Each container's identity (image digest,
container ID, hostname) is captured into the attestation's PROV provenance —
so the RTM records *where* and *how* each piece of evidence was generated.

```bash
make compute-build                                        # one-time image build
uv run python -m pipeline.runner --auto --compute=docker  # remote-emulated compute
```

Per-activity provenance after a Docker run:

```turtle
adcs:SA-REQ-003
    prov:atLocation        <urn:adcs:location:docker:71a59f23f3e9> ;
    prov:wasAssociatedWith <urn:adcs:executor:71a59f23f3e9> ;
    prov:startedAtTime     "2026-05-14T02:27:51Z"^^xsd:dateTime ;
    prov:endedAtTime       "2026-05-14T02:27:56Z"^^xsd:dateTime .

<urn:adcs:executor:71a59f23f3e9> a prov:SoftwareAgent ;
    rtm:hostname      "71a59f23f3e9" ;
    rtm:imageDigest   "sha256:92bb8bf18f5f..." ;
    rtm:imageLabel    "adcs-compute:latest" ;
    rtm:containerId   "71a59f23f3e9" ;
    rtm:pythonVersion "3.12.13" .
```

## Toolchain

| What                       | Required?    | Used for                                                 |
| -------------------------- | ------------ | -------------------------------------------------------- |
| Python 3.12, uv            | yes          | runtime + tests + ontology build                         |
| Docker                     | optional     | `--compute=docker` Stage 2/3 emulation                   |
| OBO ROBOT (Java JAR)       | optional     | `make ontology-robot` (ELK reasoning + OBO hygiene)      |
| `FLEXO_TOKEN` env var      | optional     | `--backend=flexo` live push                              |

`uv run python -m pipeline.runner` works with no optional tools installed.

## Ontology Authoring

The canonical artifact `ontology/rtm.ttl` is built. Edit
`ontology/rtm-edit.ttl` and rebuild:

```bash
make fetch-imports        # one-time: pull vendored upstream ontologies
make ontology             # Python-only build (default, no Java needed)
make ontology-robot       # optional: ROBOT merge + ELK reason + report (needs Java + obo-robot)
```

The build validates every upstream term `rtm-edit.ttl` references exists in
the vendored copy and regenerates `assembly_manifest.json`. The Stage 0 banner
is data-driven from that manifest.

## Key Directories

- [`ontology/`](ontology/) — TBox + shapes + vendored imports + build manifest
- [`structural/`](structural/) — SysMLv2 RDF (satellite.ttl, parameters.ttl)
- [`analysis/`](analysis/) — SymPy proofs + scipy simulation
- [`evidence/`](evidence/) — content hashing + RDF binding (with execution provenance)
- [`traceability/`](traceability/) — RTM assembly, queries, attestation, validation, audit
- [`pipeline/`](pipeline/) — stage orchestrator + Dataset helpers + plan.ttl + backends
- [`interrogate/`](interrogate/) — explain / reproduce / visualize
- [`compute/`](compute/) — Local + Docker compute backends + Dockerfile
- [`flexo/`](flexo/) — Flexo MMS provisioning scripts + live integration docs
- [`scripts/`](scripts/) — `fetch_imports.py` + `build_ontology.py`
- [`tests/`](tests/) — 166 tests across ontology, named graphs, shapes, audit, backends, compute, live Flexo

## Future Work

The demo deliberately stops short of several production-grade extensions.
Each is documented in [the plan file](/Users/z/.claude/plans/i-want-to-look-hidden-balloon.md)
and summarized here so it's easy to pick up:

- **Cryptographic envelopes & signatures.** Today's hashes are bare
  SHA-256: content identity but not authenticity. Production should
  layer W3C VC Data Integrity (Ed25519 over RDF canonicalization),
  in-toto/SLSA build attestations, and sigstore/Rekor transparency logs.
- **Formal authority & credential model.** FOAF + W3C Org Ontology +
  `schema:hasCredential` + W3C Verifiable Credentials layered on top
  of `prov:Agent` — so an attestation records who attested, what role
  they were authorized in, and what credentials backed that authority.
- **OntoGSN confidence arguments.** Reify confidence in each
  Assumption / Justification node so "how confident are you in the
  adequacy claim?" is a queryable graph instead of prose.
- **Defeaters & revocation** (SACM-style) — invalidate attestations
  when later evidence contradicts an earlier assumption.
- **Multi-attestation aggregation policy.** Sign-off gates
  (Engineering + QA + Certifier must all attest `earl:passed`)
  expressed as SHACL constraints on requirement transitions.
- **Production Flexo deployment.** Multi-user SSO auth, PR-style
  branches for RTM evolution, CI hooks, federation across program-
  level Flexo instances.
- **OSLC connector** for DOORS Next / Jama / RQM, on top of the
  alignment axioms already in place.
- **Federated SPARQL** for cross-program traceability when satellite-
  and ADCS-level requirements live in different repositories.
- **Continuous re-verification in CI** plus a **live traceability
  dashboard** — both compose with the cryptographic-envelope work.

## License

Apache 2.0 — see [LICENSE](LICENSE).
