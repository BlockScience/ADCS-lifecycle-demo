# ============================================================================
# ADCS Lifecycle Demo — ontology build chain
#
# Two paths (WP2 §4.A):
#   - Default (REQUIRES Java + obo-robot on PATH): `make ontology`
#       Runs the Python assembly (refs check, equivalence axioms,
#       rtm.ttl + assembly_manifest.json) and then ROBOT/ELK
#       verification (merge against catalog, ELK reasoning, OBO
#       hygiene report). Fails fast when the toolchain is missing.
#       This is the canonical build path. CI runs it on every push.
#   - No-Java alternative (explicit, named): `make ontology-python`
#       Runs only the Python assembly. The integration story does NOT
#       silently degrade — invoking this target is an intentional
#       opt-out, not a flag on the default. The manifest records
#       `robot_used: false` so Stage 0 reports the difference.
#
# Demo users running the pipeline (`uv run python -m pipeline.runner`)
# do NOT need either path — rtm.ttl is committed.
#
# ROBOT note: macOS's `brew install robot-framework` installs Robot
# Framework (Python test tool) which owns /usr/local/bin/robot. The OBO
# ROBOT is installed as `obo-robot` to avoid the name collision. Set
# the ROBOT variable below if your install uses a different name.
# ============================================================================

ROBOT ?= obo-robot
JAVA_HOME ?= /usr/local/opt/openjdk

.PHONY: help fetch-imports ontology ontology-python ontology-robot _robot-preflight _ontology-python-stamp validate-imports clean-imports flexo-up flexo-init flexo-down

help:
	@echo "Ontology build targets:"
	@echo "  fetch-imports     Pull upstream ontologies into ontology/imports/"
	@echo "  ontology          Canonical build: Python assembly + ROBOT/ELK verification (requires Java + obo-robot)"
	@echo "  ontology-python   Python assembly only (no-Java path; ROBOT verification skipped)"
	@echo "  ontology-robot    Run ROBOT merge + ELK reason + report against rtm-edit.ttl (no rtm.ttl rewrite)"
	@echo "  validate-imports  Check vendored imports parse and contain referenced terms"
	@echo "  clean-imports     Remove ontology/imports/ (forces re-fetch)"
	@echo ""
	@echo "Flexo MMS targets (Phase J — not yet wired):"
	@echo "  flexo-up          Bring up local Flexo + Fuseki via Docker Compose"
	@echo "  flexo-init        Provision adcs-demo org/project/repo/branch in Flexo"
	@echo "  flexo-down        Tear down the Flexo stack"

fetch-imports:
	uv run python -m scripts.fetch_imports

# Default canonical build. Fails fast if Java/ROBOT are missing. The
# preflight + ROBOT verification run BEFORE the stamped Python build so
# the manifest's `robot_used: true` is only written after ROBOT clears.
ontology: _robot-preflight ontology-robot _ontology-python-stamp
	@echo "[ontology] PASS — Python assembly + ROBOT/ELK verification clean."

# No-Java path. Explicit, named target. Manifest records
# `robot_used: false`; Stage 0 prints the Python-only banner.
ontology-python:
	uv run python -m scripts.build_ontology

# Internal: runs the Python assembly with ADCS_ROBOT_VERIFIED=1 so the
# manifest records `robot_used: true`. Called from `ontology` after the
# ROBOT verification step has passed.
_ontology-python-stamp:
	ADCS_ROBOT_VERIFIED=1 uv run python -m scripts.build_ontology

validate-imports:
	@uv run python -c "from rdflib import Graph; \
import sys; \
from pathlib import Path; \
ok = True; \
[print(f'  OK  {p.name}: {len(Graph().parse(p, format=\"turtle\"))} triples') for p in Path('ontology/imports').glob('*.ttl')]"

# Preflight: fail-fast on missing Java or obo-robot. The no-Java escape
# hatch is the explicit `make ontology-python` target (not a flag).
_robot-preflight:
	@command -v $(ROBOT) >/dev/null 2>&1 || { \
		echo ""; \
		echo "ERROR: ROBOT not found on PATH (looking for: $(ROBOT))."; \
		echo ""; \
		echo "  ROBOT/ELK is the default verifier for the assembled ontology."; \
		echo "  Install obo-robot (http://robot.obolibrary.org) or override the"; \
		echo "  command name: ROBOT=/path/to/robot make ontology"; \
		echo ""; \
		echo "  No-Java path (explicit, separate target):"; \
		echo "    make ontology-python"; \
		echo ""; \
		exit 1; \
	}
	@PATH="$(JAVA_HOME)/bin:$$PATH" $(ROBOT) --version >/dev/null 2>&1 || { \
		echo ""; \
		echo "ERROR: ROBOT found but Java is missing or unreachable."; \
		echo "  Set JAVA_HOME (currently: $(JAVA_HOME)) to a JRE 11+ install,"; \
		echo "  or use the no-Java path: make ontology-python"; \
		echo ""; \
		exit 1; \
	}

ontology-robot: _robot-preflight
	@echo "[ROBOT] Merging rtm-edit.ttl + vendored imports (via ontology/catalog-v001.xml) ..."
	PATH="$(JAVA_HOME)/bin:$$PATH" $(ROBOT) merge \
		--catalog ontology/catalog-v001.xml \
		--input ontology/rtm-edit.ttl \
		--output ontology/rtm-robot-merged.ttl
	@echo "[ROBOT] Reasoning (ELK) — consistency check ..."
	PATH="$(JAVA_HOME)/bin:$$PATH" $(ROBOT) reason \
		--reasoner ELK \
		--input ontology/rtm-robot-merged.ttl \
		--output ontology/rtm-robot-reasoned.ttl
	@echo "[ROBOT] Report (OBO hygiene lints — profiled to scope rules to our terms) ..."
	PATH="$(JAVA_HOME)/bin:$$PATH" $(ROBOT) report \
		--input ontology/rtm-robot-reasoned.ttl \
		--profile ontology/robot-report-profile.txt \
		--output ontology/rtm-robot-report.tsv \
		--fail-on ERROR \
		|| { echo "ROBOT report flagged ERROR-level findings; see ontology/rtm-robot-report.tsv"; exit 1; }
	@echo ""
	@echo "[ROBOT] Validation complete."
	@echo "  Merged-with-imports:  ontology/rtm-robot-merged.ttl"
	@echo "  Reasoned:             ontology/rtm-robot-reasoned.ttl"
	@echo "  Report:               ontology/rtm-robot-report.tsv"
	@echo ""
	@echo "  Canonical artifact (ontology/rtm.ttl) is unchanged — produced by 'make ontology'."
	@echo "  ROBOT outputs are validation/inspection artifacts (gitignored)."

clean-imports:
	rm -rf ontology/imports/

# ----------------------------------------------------------------------------
# Flexo MMS targets
#
# Default target is the remote starforge instance (FLEXO_URL=
# https://try-layer1.starforge.app), set FLEXO_TOKEN to a bearer token
# you obtained from a collaborator. For offline work, use the local
# Compose stack from openmbee/flexo-mms-deployment and override
# FLEXO_URL=http://localhost:8080.
# ----------------------------------------------------------------------------

flexo-init:
	./flexo/init.sh

flexo-down:
	./flexo/teardown.sh

# Live pipeline run against whichever Flexo target FLEXO_URL points at.
flexo-run:
	uv run python -m pipeline.runner --auto --backend=flexo

# ----------------------------------------------------------------------------
# Docker-emulated remote compute (Phase L)
#
# Emulates a remote analysis server for Stage 2 / Stage 3. The provenance
# captured (image digest, container ID, container hostname) lands in the
# attestation graph so the RTM records WHERE and HOW each piece of
# evidence was produced.
# ----------------------------------------------------------------------------

COMPUTE_IMAGE ?= adcs-compute:latest

.PHONY: compute-build compute-run compute-shell

compute-build:
	docker build -t $(COMPUTE_IMAGE) -f compute/Dockerfile .

compute-run:
	uv run python -m pipeline.runner --auto --compute=docker

compute-shell:
	docker run --rm -it -v $$(pwd):/work -w /work $(COMPUTE_IMAGE) bash
