"""
Reproducibility utilities for the RajNLP-50K training pipeline.

Provides a single canonical ``set_all_seeds`` function that fixes all
pseudo-random number generators used across the project (Python ``random``,
NumPy, and PyTorch) before any stochastic operation.

Requirements: 17.1
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    """Fix random seeds for Python ``random``, NumPy, and PyTorch.

    Sets:
    - ``random.seed(seed)``
    - ``numpy.random.seed(seed)`` (if NumPy is available)
    - ``torch.manual_seed(seed)`` (if PyTorch is available)
    - ``torch.cuda.manual_seed_all(seed)`` (if CUDA is available)

    The seed value is logged at INFO level to the experiment log.

    Call this function at the start of every training script before any
    stochastic operation to satisfy Requirement 17.1.

    Args:
        seed: Integer seed value.  Must be non-negative.

    Requirements: 17.1
    """
    import random

    random.seed(seed)
    logger.info("Set Python random seed: %d", seed)

    # NumPy — graceful skip if not installed
    try:
        import numpy as np  # type: ignore[import]

        np.random.seed(seed)
        logger.info("Set NumPy random seed: %d", seed)
    except ImportError:
        logger.warning("NumPy not available; skipping NumPy seed setting")

    # PyTorch — graceful skip if not installed
    try:
        import torch  # type: ignore[import]

        torch.manual_seed(seed)
        logger.info("Set PyTorch manual seed: %d", seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            logger.info("Set PyTorch CUDA seed: %d", seed)
    except ImportError:
        logger.warning("PyTorch not available; skipping PyTorch seed setting")
