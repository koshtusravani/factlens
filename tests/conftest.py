"""Shared pytest fixtures. All tests run on tiny files in tests/fixtures; no downloads, no GPU."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factlens.config import load_config  # noqa: E402
from factlens.data.fever import load_claims  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def claims():
    return load_claims(FIXTURES / "fever_mini.jsonl")


@pytest.fixture(scope="session")
def cfg():
    return load_config()