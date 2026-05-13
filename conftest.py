"""
Root-level pytest configuration for the RajNLP-50K project.

This file is automatically loaded by pytest before any tests are collected.
It configures Hypothesis settings profiles for property-based testing.
"""

from hypothesis import HealthCheck, settings

# ---------------------------------------------------------------------------
# Hypothesis settings profiles
# ---------------------------------------------------------------------------

# Default profile: 100 examples per property test (as specified in the design doc)
settings.register_profile(
    "default",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)

# CI profile: fewer examples for faster feedback in continuous integration
settings.register_profile(
    "ci",
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
)

# Thorough profile: more examples for deeper exploration
settings.register_profile(
    "thorough",
    max_examples=500,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.load_profile("default")
