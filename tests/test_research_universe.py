"""Tests for symbol universe definitions."""

from __future__ import annotations

import pytest

from predictor.research.universe import (
    broad_nse_universe,
    core_research_universe,
    nifty50_universe,
    test_fixture_universe as fixture_universe,
)


def test_nifty50_universe_has_expected_count() -> None:
    syms = nifty50_universe()
    assert len(syms) == 50


def test_nifty50_universe_contains_anchors() -> None:
    syms = set(nifty50_universe())
    assert "RELIANCE" in syms
    assert "TCS" in syms
    assert "INFY" in syms


def test_broad_universe_is_superset_of_nifty50() -> None:
    nifty = set(nifty50_universe())
    broad = set(broad_nse_universe())
    assert nifty.issubset(broad)
    assert len(broad) > len(nifty)


def test_core_research_universe_has_no_duplicates() -> None:
    syms = core_research_universe()
    assert len(syms) == len(set(syms))


def test_core_research_universe_covers_multiple_sectors() -> None:
    syms = core_research_universe()
    # Spot checks for sectoral breadth
    assert "RELIANCE" in syms      # energy / conglomerate
    assert "HDFCBANK" in syms      # banking
    assert "INFY" in syms          # IT
    assert "TATASTEEL" in syms     # metals
    assert "BRITANNIA" in syms     # FMCG defensive


def test_fixture_universe_returns_known_files() -> None:
    syms = fixture_universe()
    assert "INFY_NS_1d" in syms
    assert "TCS_NS_1d" in syms
    assert len(syms) >= 4
    assert isinstance(syms, tuple)


def test_all_universes_return_tuples_of_strings() -> None:
    for fn in (nifty50_universe, broad_nse_universe, core_research_universe, fixture_universe):
        result = fn()
        assert isinstance(result, tuple)
        assert all(isinstance(s, str) for s in result)
