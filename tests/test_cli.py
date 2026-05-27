"""Smoke tests for Typer CLI surfaces (WP1 §4.6).

Uses typer.testing.CliRunner (re-exported from Click). For each app:
- `--help` exits 0 and lists every documented flag
- Enum-typed options reject unknown values with non-zero exit

The full pipeline run is exercised by tests/test_pipeline.py; the
tests here only verify the CLI shell.
"""

from __future__ import annotations

from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# pipeline.runner — Typer-migrated from argparse in WP1 §4.6
# ---------------------------------------------------------------------------

def test_pipeline_runner_help_lists_known_flags():
    """--help renders Typer-style output and lists every documented flag."""
    from pipeline.runner import app
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for flag in (
        "--auto", "--no-attest", "--engineer",
        "--rebuild", "--backend", "--compute",
    ):
        assert flag in result.stdout, (
            f"Missing {flag} in --help output:\n{result.stdout}"
        )


def test_pipeline_runner_rejects_unknown_backend():
    """Enum-typed --backend choice rejects values outside the enum."""
    from pipeline.runner import app
    result = runner.invoke(app, ["--backend", "bogus"])
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "bogus" in combined or "Invalid value" in combined


def test_pipeline_runner_rejects_unknown_compute():
    """Enum-typed --compute choice rejects values outside the enum."""
    from pipeline.runner import app
    result = runner.invoke(app, ["--compute", "bogus"])
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "bogus" in combined or "Invalid value" in combined


def test_pipeline_runner_main_callable_preserved_for_console_script():
    """`[project.scripts] adcs-pipeline = "pipeline.runner:main"` requires
    the `main` symbol to remain importable + callable."""
    from pipeline.runner import main
    assert callable(main)
