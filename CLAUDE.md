# ADCS Lifecycle Demo

## What this is

A demonstration of bidirectional requirements traceability for a satellite
Attitude Determination and Control System (ADCS). The demo traces the full
lifecycle from SysMLv2 structural specification through symbolic analysis
(SymPy), numerical simulation (scipy), and human expert attestation — with
all traceability stored as RDF in a named-graph quadstore and all artifacts
version-controlled in git.

## Core principle

**Evidence does not verify requirements; evidence supports a human judgment
that requirements are satisfied.** Models are imperfect representations of
physical systems. The engineer judges model adequacy and evidence sufficiency.
Only human attestation connects evidence to requirement satisfaction.

This split is the Hawkins–Habli Assurance Claim Point categorization:
adequacy is a `gsn:Assumption`, sufficiency is a `gsn:Justification`. The
project's `rtm:` namespace introduces NO novel epistemic vocabulary.

## Architecture — integration ontology

`rtm:` is a thin application/integration ontology. Convenience classes and
properties are subclasses / subproperties of established standards; the
project's contribution is the assembly, not new terms.

| Namespace                            | Role                                             |
| ------------------------------------ | ------------------------------------------------ |
| `prov:`                              | Provenance spine                                 |
| `sysml:` ↔ `omg-sysml:`              | SysMLv2; local prefix aliased to OMG SysMLv2 OWL |
| `earl:`                              | Assertion + outcome lattice + mode               |
| `gsn:` (OntoGSN)                     | Goal / Strategy / Solution / Assumption / Justification |
| `p-plan:`                            | Process model (one Step per pipeline stage)      |
| `oslc_rm:` / `oslc_qm:`              | Tool-interop aliases for RM tools                |
| `sh:`                                | SHACL closure-rule suite                         |

## Named-graph layout (rdflib.Dataset)

The runtime is `Dataset(default_union=True)` so SPARQL queries match across
named graphs without `GRAPH` clauses. Each layer lives in its own graph,
sized to match Flexo MMS branch conventions:

```text
<rtm:ontology>        TBox + shapes + individuals
<rtm:plan>            P-PLAN process model
<adcs:structural>     SysMLv2 instance data
<adcs:context>        Stable gsn:Context / gsn:Assumption individuals
<adcs:evidence>       rtm:Evidence artifacts (incl. rtm:DockerImage under --compute=docker, WP3)
<adcs:attestations>   rtm:Attestation events
<adcs:plan-execution> p-plan:Activity instances (one per stage)
<adcs:audit>          Forward/backward/bidirectional audit summary
```

IRIs and the `NAMED_GRAPHS` dict are in [`ontology/prefixes.py`](ontology/prefixes.py).

## Key directories

- `ontology/` — `rtm-edit.ttl` (source), `rtm.ttl` (built artifact),
  `rtm_shapes.ttl` (closure-rule suite), `rtm_individuals.ttl`,
  `sysml_term_map.csv`, `assembly_manifest.json`, `imports/` (vendored upstreams)
- `structural/` — SysMLv2 RDF model
- `analysis/` — SymPy symbolic analysis + scipy numerical simulation
- `compute/` — `LocalCompute` (in-process) + `DockerCompute` (containerized
  remote-emulation with PROV provenance capture) + `Dockerfile`
- `evidence/` — content hashing + RDF evidence binding (with execution metadata)
- `traceability/` — RTM assembly, SPARQL queries, attestation (GSN-based),
  closure-rule verification (`verification.py`), audit module
- `pipeline/` — stage orchestrator (`PipelineState` + per-stage free
  functions in `runner.py`; `state.py` defines the typed result records),
  `dataset.py` (named-graph helpers incl. `query_named_graph`),
  `stage0_assembly.py`, `plan.ttl`, `backends/` (Local / Flexo / Fuseki)
- `flexo/` — Flexo MMS provisioning scripts + integration docs
- `interrogate/` — explain / reproduce / visualize / rerun
- `scripts/` — `fetch_imports.py`, `build_ontology.py`
- `tests/` — 166 tests (alignment, named graphs, shape suite, audit,
  backends, compute, live Flexo opt-in)

## Pipeline (stages 0–8)

Stage 0 narrates the ontology assembly; Stage 6.5 runs the closure-rule
suite; Stage 7a runs the audit. Every stage emits a `p-plan:Activity`
into `<adcs:plan-execution>` so the construction process is itself
queryable.

### Pipeline state + structured stage results

The orchestrator threads a [`PipelineState`](pipeline/state.py) object
through per-stage free functions (`run_stage_<N>_<name>(state) ->
<StageResult>` in [`pipeline/runner.py`](pipeline/runner.py)). Each
stage returns a frozen dataclass (`StructuralResult`,
`SymbolicResult`, `NumericalResult`, …) that the next stage reads via
`state.<prior>.<field>`. The runner's job is narration + the ordered
call sequence; stage bodies stand alone and are unit-testable.

`PipelineState.activity_to_stage` maps `p-plan` step IRI fragments
(`STEP_NAMES` in
[`traceability/plan_execution.py`](traceability/plan_execution.py))
to stage numbers; [`interrogate/rerun.py`](interrogate/rerun.py)
keeps a parallel `ACTIVITY_TO_STAGE` table, cross-checked by a unit
test against `STEP_NAMES`.

## CLI surface

Every CLI in this repo is a Typer app (Click + Rich transitively).
Pattern per module: `app = typer.Typer(...)` + `@app.command()`
function + `if __name__ == "__main__": app()`. Existing `uv run python
-m <module>` invocations and the `[project.scripts] adcs-pipeline`
console script work unchanged. Choice-validated options use `Enum`
subclasses so Typer matches the prior argparse `choices=` semantics.

CLIs today:

- `pipeline.runner` — runs the full lifecycle (flags: `--auto`,
  `--no-attest`, `--engineer`, `--rebuild`, `--backend`, `--compute`).
- `interrogate.rerun` — translates a verification report into the
  pipeline stages that must re-run (flags: `--input`,
  `--requirement`, `--format`).

`interrogate.explain`, `interrogate.reproduce`, `interrogate.visualize`
are library-only (no CLI entry points). Tests use
`typer.testing.CliRunner` in [`tests/test_cli.py`](tests/test_cli.py).
A top-level `adcs` aggregator (`adcs pipeline run`, `adcs interrogate
rerun`, etc.) is tracked as a follow-up — see
[issue #5](https://github.com/DynamicalSystemsGroup/ADCS-lifecycle-demo/issues/5).

## Verification vs validation (term discipline)

Strict semantic split across module names, function names, doc
strings, RDF property labels, log/banner strings, and commit messages:

- **Verification** = automated check whose computation result is
  fully specified (SHACL conformance, ROBOT/ELK consistency,
  content-hash matching, completeness checks, HTTP-connectivity
  probes, triple-count budgets).
- **Validation** = human judgement with expertise, additional context,
  and/or interpretation (engineer attestation, adequacy assumption,
  sufficiency justification).

This breaks with pyshacl's `validate()` convention deliberately —
upstream APIs keep their names, the demo's own wrappers use the
discipline. SHACL conformance is wrapped by `traceability.verification.verify`;
human judgement lives in `traceability.attestation.request_attestation`.
The split pairs with the longer-standing rule that **evidence does not
verify requirements; only human attestation does**: evidence and
verification are both system-internal, attestation is the human
boundary.

One IRI fragment is intentionally out-of-sync with the discipline:
`<plan/step/ValidateShapes>` in `pipeline/plan.ttl` is preserved so
already-persisted `<adcs:plan-execution>` and `<adcs:audit>` graphs
stay valid. The rdfs:label says "Verify"; the IRI rename is tracked
separately for a future Flexo migration.

## Toolchain

- Python 3.12+, managed by `uv` — required
- `rdflib` for RDF/SPARQL, `sympy` for symbolic math, `scipy` for ODE integration
- `pyshacl` for closure-rule verification (the demo's own wrapper is
  named `verify`; pyshacl's upstream API is `validate` and is wrapped,
  not renamed), `httpx` for backend HTTP, `typer` for CLI surfaces
- ProofScript/ProofBuilder pattern reimplemented from gds-proof
- Docker — optional, for `--compute=docker`
- OBO ROBOT (Java) — **required for default `make ontology`**; no-Java
  users invoke the explicit `make ontology-python` target instead.
  CI installs Java 17 + a cached `robot.jar` and verifies every push.

## Running

```bash
uv run python -m pipeline.runner --auto            # local backend, local compute
uv run python -m pipeline.runner --auto --backend=flexo   # push to Flexo MMS
uv run python -m pipeline.runner --auto --compute=docker  # remote-compute emulation
```

The runner runs against the committed `rtm.ttl`; it does NOT need
Java/obo-robot. Only **rebuilding** the ontology does.

## Tests

```bash
uv run pytest               # default: skips live + network markers
uv run pytest -m live       # opt-in: live Flexo MMS round-trip (needs FLEXO_TOKEN)
uv run pytest -m network    # opt-in: reserved for W3C-vocab fetches
```

Marker filtering is set in `pyproject.toml` via `addopts = "-m 'not
live and not network'"`. `tests/test_flexo_live.py` carries the
`live` marker; tests there fail loudly when `-m live` is requested
without credentials rather than silently skipping (skip-on-opt-in
would hide infra breakage).

## Ontology rebuild

```bash
make ontology          # canonical: Python assembly + ROBOT/ELK verification (requires Java + obo-robot)
make ontology-python   # no-Java path: Python assembly only, ROBOT verification skipped
make ontology-robot    # just the ROBOT step (no rtm.ttl rewrite)
```

`make ontology` fails fast on missing Java/obo-robot. The no-Java
escape is the explicit `ontology-python` target — invoking it is an
intentional opt-out, not a flag. The manifest's `robot_used` field +
Stage 0 banner record which path produced the artifact. The build
also enforces a triple-count budget (`TRIPLE_BUDGET=356` in
`scripts/build_ontology.py`) so the integration ontology can't quietly
grow novel epistemic vocabulary.

`rtm.ttl` is a built artifact. Edit `rtm-edit.ttl` and rebuild.

## Future work (priority order)

Captured in the plan file at `/Users/z/.claude/plans/i-want-to-look-hidden-balloon.md`:

1. **Cryptographic envelopes & signatures** — W3C VC Data Integrity
   (eddsa-rdfc-2022) + in-toto/SLSA + sigstore/Rekor. Today's bare
   SHA-256 gives content identity, not authenticity.
2. **Formal authority/credential model** — FOAF + Org Ontology +
   `schema:hasCredential` + W3C Verifiable Credentials on top of
   `prov:Agent`.
3. **OntoGSN confidence arguments** — reify per-ACP confidence.
4. **Defeaters & revocation** (SACM-style).
5. **Multi-attestation aggregation policy** — sign-off gates as SHACL.
6. **Production Flexo deployment** — multi-user auth, PR branches for
   RTM evolution, federation.
7. **OSLC connector** for DOORS Next / Jama / RQM.
8. **Federated SPARQL** for cross-program traceability.
9. **Continuous re-verification in CI** + **live traceability dashboard.**

Notebook Act 9–10 demonstrate the *capture* side of remote compute
provenance. Production replay against a pulled image is a future
addition that pairs with item #1.
