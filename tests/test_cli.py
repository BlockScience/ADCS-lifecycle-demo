"""Smoke tests for Typer CLI surfaces (WP1 §4.6).

Uses typer.testing.CliRunner (re-exported from Click). For each app:
- `--help` exits 0 and lists every documented flag
- Enum-typed options reject unknown values with non-zero exit

The full pipeline run is exercised by tests/test_pipeline.py; the
tests here only verify the CLI shell.
"""

from __future__ import annotations

import re

from typer.testing import CliRunner

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _flatten_help(text: str) -> str:
    """Strip ANSI escape codes + collapse whitespace.

    Rich's Typer help renderer inserts ANSI color codes and wraps flag
    columns to terminal width. CI runners have a narrower default
    width than dev workstations, which can split a flag across a wrap
    boundary and break naive substring matches. Flattening makes the
    substring check terminal-width-agnostic.
    """
    return re.sub(r"\s+", " ", _ANSI.sub("", text))


# ---------------------------------------------------------------------------
# pipeline.runner — Typer-migrated from argparse in WP1 §4.6
# ---------------------------------------------------------------------------

def test_pipeline_runner_help_lists_known_flags():
    """--help renders Typer-style output and lists every documented flag."""
    from pipeline.runner import app
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    flat = _flatten_help(result.stdout)
    for flag in (
        "--auto", "--no-attest", "--engineer",
        "--rebuild", "--backend", "--compute",
    ):
        assert flag in flat, (
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


# ---------------------------------------------------------------------------
# interrogate.rerun — new Typer CLI introduced in WP1 §4.5 (closes #3)
# ---------------------------------------------------------------------------

def test_rerun_help_lists_known_flags():
    from interrogate.rerun import app
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    flat = _flatten_help(result.stdout)
    for flag in ("--input", "--requirement", "--format"):
        assert flag in flat, (
            f"Missing {flag} in rerun --help:\n{result.stdout}"
        )


def test_rerun_returns_exit_2_when_input_missing(tmp_path):
    from interrogate.rerun import app
    bogus = tmp_path / "does-not-exist.trig"
    result = runner.invoke(app, ["--input", str(bogus)])
    assert result.exit_code == 2


def test_rerun_rejects_unknown_format(tmp_path):
    from interrogate.rerun import app
    bogus = tmp_path / "x.trig"
    bogus.write_text("")
    result = runner.invoke(app, ["--input", str(bogus), "--format", "bogus"])
    assert result.exit_code != 0
