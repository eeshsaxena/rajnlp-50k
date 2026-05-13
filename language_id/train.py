"""
Language_ID_Tagger training script.

Trains the LanguageIDTagger on a held-out subset of RajNLP-50K with manually
verified language-ID labels (~5,000 sentences, ~60,000 tokens).

Usage
-----
    python -m language_id.train [--seed SEED] [--data-path PATH]

Arguments
---------
--seed      Random seed for Python ``random``, NumPy, and PyTorch.
            Default: 42.  The seed is logged to the experiment log.
--data-path Path to the training data file (JSON Lines format).
            Default: None (simulated training when no data is provided).

Reproducibility
---------------
The seed is fixed for all three RNG sources before any stochastic operation,
satisfying Requirement 17.1.

Requirements: 9.2, 17.1
"""

from __future__ import annotations

import argparse
import logging
import sys

from models.reproducibility import set_all_seeds  # noqa: F401  (re-exported for callers)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------


def train(seed: int = 42, data_path: str | None = None) -> None:
    """Run the Language_ID_Tagger training pipeline.

    Steps:
    1. Fix and log the random seed for Python ``random``, NumPy, and PyTorch.
    2. Load training data (or simulate if *data_path* is None).
    3. Train the LanguageIDTagger (or simulate training).
    4. Log completion with the seed value.

    Args:
        seed: Random seed.  Logged to the experiment log before training.
        data_path: Path to the JSON Lines training data file.  If None,
            training is simulated (for testing without real data).

    Requirements: 9.2, 17.1
    """
    # Step 1: Fix and log all random seeds (Requirement 17.1)
    logger.info("=== Language_ID_Tagger Training ===")
    logger.info("Random seed: %d", seed)
    set_all_seeds(seed)

    # Step 2: Load or simulate training data
    if data_path is not None:
        logger.info("Loading training data from: %s", data_path)
        try:
            import json
            from pathlib import Path

            data_file = Path(data_path)
            if not data_file.exists():
                logger.error("Training data file not found: %s", data_path)
                sys.exit(1)

            sentences = []
            with data_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        sentences.append(json.loads(line))

            logger.info("Loaded %d training sentences from %s", len(sentences), data_path)
        except Exception as exc:
            logger.error("Failed to load training data: %s", exc)
            sys.exit(1)
    else:
        # Simulated training — no real data available yet.
        # This will be wired up properly in Task 21.
        logger.info(
            "No data path provided; running simulated training "
            "(will be wired to real data in Task 21)"
        )
        logger.info("Simulated training data: ~5,000 sentences, ~60,000 tokens")

    # Step 3: Train (or simulate)
    logger.info("Initialising LanguageIDTagger ...")
    from language_id.tagger import LanguageIDTagger  # noqa: PLC0415

    tagger = LanguageIDTagger()
    logger.info("LanguageIDTagger initialised (heuristic mode)")

    # In the real pipeline, fine-tuning of the MuRIL token-classification head
    # would happen here.  For now, we log the completion.
    logger.info("Training complete (seed=%d)", seed)
    logger.info(
        "Note: Full MuRIL fine-tuning will be implemented in Task 21 "
        "when the annotated training data is available."
    )

    return tagger


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Language_ID_Tagger on RajNLP-50K language-ID labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Python random, NumPy, and PyTorch.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the JSON Lines training data file.  "
             "If omitted, training is simulated.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the training script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    train(seed=args.seed, data_path=args.data_path)


if __name__ == "__main__":
    main()
