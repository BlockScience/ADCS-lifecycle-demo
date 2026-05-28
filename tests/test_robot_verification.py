"""Phase B — opt-in tests that validate the OBO ROBOT path works.

These tests SKIP cleanly when ROBOT or Java are not available, so CI
without those tools (and demo users running just `uv run pytest`) are
not blocked. They run when:
  - `obo-robot --version` (or `ROBOT_CMD=...`) succeeds, AND
  - the JAVA_HOME / PATH environment makes Java reachable.

The point of these tests is to confirm `make ontology-robot` validates
our integration ontology cleanly:
  - All equivalence axioms parse
  - ELK accepts the merged ontology as consistent
  - The OBO hygiene report (with our profile) emits zero ERROR findings
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ROBOT_CMD = os.environ.get("ROBOT_CMD", "obo-robot")
JAVA_HOME = os.environ.get("JAVA_HOME", "/usr/local/opt/openjdk")


def _robot_available() -> bool:
    """ROBOT available and Java reachable through it."""
    if shutil.which(ROBOT_CMD) is None:
        return False
    env = {**os.environ, "PATH": f"{JAVA_HOME}/bin:{os.environ.get('PATH', '')}"}
    try:
        proc = subprocess.run(
            [ROBOT_CMD, "--version"], env=env, capture_output=True, timeout=20
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


pytestmark = pytest.mark.skipif(
    not _robot_available(),
    reason="OBO ROBOT or Java unavailable; ROBOT validation is opt-in",
)


def test_make_ontology_robot_succeeds():
    """Drive the full Make target. Asserts merge + reason + report (with
    our scoped profile) all succeed end-to-end."""
    proc = subprocess.run(
        ["make", "ontology-robot"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"make ontology-robot failed (rc={proc.returncode}).\n"
            f"stdout (last 40 lines):\n{chr(10).join(proc.stdout.splitlines()[-40:])}\n"
            f"stderr (last 40 lines):\n{chr(10).join(proc.stderr.splitlines()[-40:])}"
        )


def test_robot_artifacts_present():
    """After `make ontology-robot`, three output files should exist."""
    for filename in (
        "rtm-robot-merged.ttl",
        "rtm-robot-reasoned.ttl",
        "rtm-robot-report.tsv",
    ):
        path = ROOT / "ontology" / filename
        assert path.exists(), f"ROBOT artifact missing: {path.relative_to(ROOT)}"
        assert path.stat().st_size > 0, f"ROBOT artifact empty: {path.relative_to(ROOT)}"


def test_robot_report_zero_errors():
    """The OBO hygiene report should emit no ERROR-level findings under
    our profile. WARN-level findings are acceptable (upstream-inherited)."""
    report = ROOT / "ontology" / "rtm-robot-report.tsv"
    if not report.exists():
        pytest.skip("Report not present; run `make ontology-robot` first")
    lines = report.read_text().splitlines()
    error_lines = [ln for ln in lines if ln.startswith("ERROR")]
    assert not error_lines, (
        f"ROBOT report has {len(error_lines)} ERROR-level findings:\n"
        + "\n".join(error_lines[:10])
    )


def test_robot_merged_richer_than_lean_artifact():
    """The merged-with-imports artifact should have substantially more
    triples than the lean canonical rtm.ttl (sanity check that imports
    actually resolved via the catalog)."""
    from rdflib import Graph

    lean = Graph()
    lean.parse(ROOT / "ontology" / "rtm.ttl", format="turtle")

    merged_path = ROOT / "ontology" / "rtm-robot-merged.ttl"
    if not merged_path.exists():
        pytest.skip("Merged artifact not present; run `make ontology-robot` first")
    merged = Graph()
    merged.parse(merged_path, format="turtle")

    # Canonical lean artifact is just our integration glue (~150 triples).
    # Merged should include PROV-O, EARL, OntoGSN, etc. — thousands of triples.
    assert len(merged) >= len(lean) * 10, (
        f"Merged artifact ({len(merged)} triples) doesn't appear to include the "
        f"vendored imports; lean is {len(lean)}. Catalog may not be resolving."
    )
