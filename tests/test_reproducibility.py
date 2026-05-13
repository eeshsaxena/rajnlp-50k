"""
Unit tests for models.reproducibility.set_all_seeds.

Validates that:
- Python's ``random`` module is seeded correctly (same seed → same sequence)
- NumPy's RNG is seeded correctly (same seed → same sequence)
- The seed value is logged at INFO level
- Different seeds produce different sequences
- Same seed produces identical sequences (core reproducibility test)

Requirements: 17.1, 17.5
"""

from __future__ import annotations

import logging
import random

import numpy as np
import pytest

from models.reproducibility import set_all_seeds


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _generate_python_random_floats(n: int = 10) -> list[float]:
    """Return *n* floats from Python's ``random`` module."""
    return [random.random() for _ in range(n)]


def _generate_numpy_random_floats(n: int = 10) -> list[float]:
    """Return *n* floats from NumPy's default RNG."""
    return list(np.random.random(n))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSetAllSeedsPythonRandom:
    """Verify that set_all_seeds seeds Python's random module."""

    def test_set_all_seeds_sets_python_random(self) -> None:
        """Same seed must produce the same Python random sequence."""
        set_all_seeds(42)
        seq1 = _generate_python_random_floats()

        set_all_seeds(42)
        seq2 = _generate_python_random_floats()

        assert seq1 == seq2, (
            "Python random sequences differ despite using the same seed"
        )

    def test_python_random_sequence_is_deterministic_across_calls(self) -> None:
        """Calling set_all_seeds resets the state, not just advances it."""
        set_all_seeds(99)
        first_value_a = random.random()

        set_all_seeds(99)
        first_value_b = random.random()

        assert first_value_a == first_value_b


class TestSetAllSeedsNumpyRandom:
    """Verify that set_all_seeds seeds NumPy's random module."""

    def test_set_all_seeds_sets_numpy_random(self) -> None:
        """Same seed must produce the same NumPy random sequence."""
        set_all_seeds(42)
        seq1 = _generate_numpy_random_floats()

        set_all_seeds(42)
        seq2 = _generate_numpy_random_floats()

        assert seq1 == seq2, (
            "NumPy random sequences differ despite using the same seed"
        )

    def test_numpy_random_sequence_is_deterministic_across_calls(self) -> None:
        """Calling set_all_seeds resets NumPy state."""
        set_all_seeds(7)
        first_value_a = float(np.random.random())

        set_all_seeds(7)
        first_value_b = float(np.random.random())

        assert first_value_a == first_value_b


class TestSetAllSeedsLogging:
    """Verify that set_all_seeds logs the seed value at INFO level."""

    def test_set_all_seeds_logs_seed_value(self, caplog: pytest.LogCaptureFixture) -> None:
        """The seed integer must appear in an INFO-level log record."""
        seed = 12345
        with caplog.at_level(logging.INFO, logger="models.reproducibility"):
            set_all_seeds(seed)

        # At least one log record should mention the seed value
        seed_str = str(seed)
        matching = [r for r in caplog.records if seed_str in r.getMessage()]
        assert matching, (
            f"Expected at least one INFO log record containing '{seed_str}', "
            f"but got: {[r.getMessage() for r in caplog.records]}"
        )

    def test_set_all_seeds_logs_at_info_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """Log records must be at INFO level (not DEBUG or WARNING)."""
        with caplog.at_level(logging.INFO, logger="models.reproducibility"):
            set_all_seeds(1)

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert info_records, "Expected at least one INFO-level log record from set_all_seeds"


class TestDifferentSeedsDifferentSequences:
    """Verify that different seeds produce different sequences."""

    def test_different_seeds_produce_different_sequences(self) -> None:
        """Two different seeds must not produce the same random sequence."""
        set_all_seeds(1)
        seq_seed1 = _generate_python_random_floats()

        set_all_seeds(2)
        seq_seed2 = _generate_python_random_floats()

        assert seq_seed1 != seq_seed2, (
            "Different seeds produced identical Python random sequences — "
            "this is astronomically unlikely and indicates a bug"
        )

    def test_different_seeds_produce_different_numpy_sequences(self) -> None:
        """Two different seeds must not produce the same NumPy sequence."""
        set_all_seeds(100)
        seq1 = _generate_numpy_random_floats()

        set_all_seeds(200)
        seq2 = _generate_numpy_random_floats()

        assert seq1 != seq2, (
            "Different seeds produced identical NumPy random sequences"
        )


class TestSameSeedIdenticalSequences:
    """Core reproducibility test: same seed → identical sequences."""

    def test_same_seed_produces_identical_sequences(self) -> None:
        """
        Run two micro-simulations with the same seed and verify that the
        generated random values are identical within floating-point tolerance.

        This is the core reproducibility property (Requirement 17.5).
        """
        n = 100

        set_all_seeds(42)
        python_run1 = [random.random() for _ in range(n)]
        numpy_run1 = list(np.random.random(n))

        set_all_seeds(42)
        python_run2 = [random.random() for _ in range(n)]
        numpy_run2 = list(np.random.random(n))

        # Python random: exact equality (deterministic integer arithmetic)
        assert python_run1 == python_run2, (
            "Python random sequences differ between two runs with the same seed"
        )

        # NumPy: within floating-point tolerance
        assert np.allclose(numpy_run1, numpy_run2, rtol=0, atol=1e-15), (
            "NumPy random sequences differ beyond floating-point tolerance "
            "between two runs with the same seed"
        )

    def test_same_seed_reproducibility_multiple_seeds(self) -> None:
        """Reproducibility holds for several different seed values."""
        for seed in (0, 1, 42, 123, 999_999):
            set_all_seeds(seed)
            seq_a = _generate_python_random_floats(20)

            set_all_seeds(seed)
            seq_b = _generate_python_random_floats(20)

            assert seq_a == seq_b, (
                f"Python random sequences differ for seed={seed}"
            )
