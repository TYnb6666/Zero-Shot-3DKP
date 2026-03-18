"""Tests for the Click CLI entry point in zerokey/cli.py."""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from zerokey.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_help_exits_zero(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0, result.output


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0, result.output
    assert "0.1.0" in result.output


def test_eval_subcommand_exists(runner: CliRunner) -> None:
    result = runner.invoke(main, ["eval", "--help"])
    assert result.exit_code == 0, result.output


def test_unknown_command_fails(runner: CliRunner) -> None:
    result = runner.invoke(main, ["nonexistent"])
    assert result.exit_code != 0
