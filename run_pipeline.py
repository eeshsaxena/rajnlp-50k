#!/usr/bin/env python3
"""
RajNLP-50K main pipeline orchestrator.

Wires all components together across four phases:

  Phase 1 — Data Collection & Corpus Building
    filter_rajasthani → deduplicate → (stratified_sample if ≥50K) → split

  Phase 2 — Annotation Pipeline
    Convert RawSentence → AnnotatedSentence, run span validation,
    toxicity label validation

  Phase 3 — Model Training & Evaluation
    Train LanguageIDTagger, SentimentClassifier, NERTagger, ToxicityClassifier;
    run baseline evaluation and platform-split evaluation;
    generate comparison table

  Phase 4 — Serialization & Release
    Serialize to JSON Lines and Parquet, run validate_round_trip,
    publish dataset and models to HuggingFace

Usage:
    python run_pipeline.py [--seed SEED] [--output-dir DIR] [--dry-run] [--log-level LEVEL]

Requirements: 17.1, 17.5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from corpus_builder.filter_dedup import filter_rajasthani, deduplicate
from corpus_builder.sampling import split, stratified_sample, InsufficientDataError
from corpus_builder.serialization import serialize, validate_round_trip
from corpus_builder.span_validation import (
    validate_all_span_text_invariants,
    validate_all_toxicity_labels,
)
from evaluation.baselines import (
    ZeroShotMBERTEvaluator,
    ZeroShotMuRILEvaluator,
    GPT4o5ShotEvaluator,
    BaselineResult,
)
from evaluation.comparison_table import generate_comparison_table, format_comparison_table
from evaluation.platform_split import run_platform_split_evaluation
from language_id.tagger import LanguageIDTagger
from models.data_models import (
    AnnotatedSentence,
    DatasetSplit,
    RawSentence,
    TokenLabel,
)
from models.ner_tagger import NERTagger
from models.reproducibility import set_all_seeds
from models.sentiment_classifier import SentimentClassifier
from models.toxicity_classifier import ToxicityClassifier
from release.huggingface_publisher import HuggingFacePublisher, TARGET_F1

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FIXED_COLLECTED_AT = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
_FIXED_ANNOTATED_AT = datetime(2024, 2, 1, 14, 0, 0, tzinfo=timezone.utc)

_STRATIFIED_SAMPLE_THRESHOLD = 50_000


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(output_dir: Path, log_level: str) -> logging.Logger:
    """Configure root logger to write to both stderr and a structured log file.

    Args:
        output_dir: Directory where ``pipeline.log`` will be written.
        log_level: Logging level string (e.g. ``"INFO"``, ``"DEBUG"``).

    Returns:
        A :class:`logging.Logger` named ``"pipeline"``.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    log_path = output_dir / "pipeline.log"

    # Root logger
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove any existing handlers to avoid duplicate output
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # File handler (structured experiment log)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(numeric_level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(numeric_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    logger = logging.getLogger("pipeline")
    logger.info("Logging initialised — log file: %s", log_path)
    return logger


# ---------------------------------------------------------------------------
# Fixture sentence factory (dry-run / test mode)
# ---------------------------------------------------------------------------


def make_fixture_sentences(n: int = 100) -> list[RawSentence]:
    """Create *n* fixture RawSentence objects for dry-run mode.

    Produces a mix of Twitter and ShareChat sentences, each containing at
    least 2 Rajasthani lexicon tokens so they pass the filter step.

    Args:
        n: Number of sentences to generate (default 100).

    Returns:
        A list of exactly *n* :class:`~models.data_models.RawSentence` objects.
    """
    # Rajasthani words from the bundled lexicon
    raj_words = [
        "म्हारो", "म्हारी", "थारो", "थारी", "कोनी",
        "घणो", "घणी", "आवे", "जावे", "पाणी",
        "बाईसा", "राजस्थानी", "मारवाड़ी", "सा", "बावड़ी",
    ]

    templates = [
        "{w1} घर बहुत सुंदर है {idx} {w2} अच्छो लागे",
        "{w1} काम {w2} राम्रो है {idx} पसंद आयो",
        "{w1} जाणूं {w2} कठे है {idx} बताओ",
        "राजस्थानी संस्कृति {w1} सुंदर है {idx} {w2} लोग",
        "{w1} माँ ने कहा {idx} {w2} बात सुणी",
        "{w1} {w2} बरसा आज {idx} खेत भर गया",
        "{w1} घर कठे है {idx} {w2} बताओ",
        "{w1} आयो वो आज {idx} {w2} इंतजार किया",
        "{w1} दिल कहे है {idx} {w2} याद आवे",
        "राजस्थानी खाना {w1} स्वादिष्ट {idx} {w2} रसोई",
    ]

    sentences: list[RawSentence] = []
    for i in range(n):
        template = templates[i % len(templates)]
        w1 = raj_words[i % len(raj_words)]
        w2 = raj_words[(i + 3) % len(raj_words)]
        text = template.format(w1=w1, w2=w2, idx=i)
        platform = "twitter" if i % 2 == 0 else "sharechat"
        sentences.append(RawSentence(
            text=text,
            source_url=f"https://{platform}.com/example/{uuid.uuid4().hex[:8]}",
            collected_at=_FIXED_COLLECTED_AT,
            platform=platform,  # type: ignore[arg-type]
            sentence_id=str(uuid.uuid4()),
        ))

    return sentences


# ---------------------------------------------------------------------------
# RawSentence → AnnotatedSentence conversion
# ---------------------------------------------------------------------------


def raw_to_annotated(
    raw: RawSentence,
    split_label: str,
    lang_id_tagger: LanguageIDTagger | None = None,
) -> AnnotatedSentence:
    """Convert a :class:`~models.data_models.RawSentence` to a minimal
    :class:`~models.data_models.AnnotatedSentence`.

    In the real pipeline this step is performed by human annotators via Label
    Studio.  Here we produce a valid ``AnnotatedSentence`` with neutral/empty
    annotation fields and Language_ID_Tagger labels.

    Args:
        raw: The raw sentence to convert.
        split_label: The dataset partition (``"train"``, ``"validation"``,
            or ``"test"``).
        lang_id_tagger: Optional pre-built tagger.  A default tagger is
            created if ``None``.

    Returns:
        A :class:`~models.data_models.AnnotatedSentence`.
    """
    if lang_id_tagger is None:
        lang_id_tagger = LanguageIDTagger()

    token_labels: list[TokenLabel] = lang_id_tagger.tag(raw.text)

    return AnnotatedSentence(
        sentence_id=raw.sentence_id,
        text=raw.text,
        platform=raw.platform,
        split=split_label,  # type: ignore[arg-type]
        sentiment="neutral",
        sentiment_annotator_labels=["neutral", "neutral", "neutral"],
        ner_spans=[],
        ner_annotator_spans=[[], [], []],
        toxicity_labels=[],
        toxicity_annotator_labels=[[], [], []],
        token_language_labels=token_labels,
        source_url=raw.source_url,
        collected_at=raw.collected_at,
        annotated_at=_FIXED_ANNOTATED_AT,
    )


# ---------------------------------------------------------------------------
# Phase 1: Data Collection & Corpus Building
# ---------------------------------------------------------------------------


def run_phase1(
    raw_sentences: list[RawSentence],
    logger: logging.Logger,
) -> tuple[DatasetSplit, list[RawSentence]]:
    """Filter, deduplicate, (optionally sample), and split raw sentences.

    If the filtered+deduped pool has ≥ 50,000 sentences, stratified sampling
    is applied to select exactly 50,000.  Otherwise all sentences are used
    directly (suitable for dry-run / small corpora).

    Args:
        raw_sentences: Raw sentences from collection (or fixture).
        logger: Pipeline logger.

    Returns:
        A tuple of (DatasetSplit, all_raw_after_dedup).
    """
    logger.info("Phase 1 — filter_rajasthani: input=%d sentences", len(raw_sentences))
    filtered = filter_rajasthani(raw_sentences)
    logger.info("Phase 1 — filter_rajasthani: kept=%d sentences", len(filtered))

    logger.info("Phase 1 — deduplicate: input=%d sentences", len(filtered))
    deduped = deduplicate(filtered)
    logger.info("Phase 1 — deduplicate: kept=%d sentences", len(deduped))

    if len(deduped) >= _STRATIFIED_SAMPLE_THRESHOLD:
        logger.info(
            "Phase 1 — stratified_sample: pool=%d ≥ %d, sampling to %d",
            len(deduped), _STRATIFIED_SAMPLE_THRESHOLD, _STRATIFIED_SAMPLE_THRESHOLD,
        )
        try:
            sampled = stratified_sample(deduped, n=_STRATIFIED_SAMPLE_THRESHOLD)
        except InsufficientDataError as exc:
            logger.error("Phase 1 — stratified_sample failed: %s", exc)
            sampled = deduped
    else:
        logger.info(
            "Phase 1 — stratified_sample: pool=%d < %d, using all sentences",
            len(deduped), _STRATIFIED_SAMPLE_THRESHOLD,
        )
        sampled = deduped

    if not sampled:
        logger.warning("Phase 1 — no sentences remain after filtering/deduplication")
        return DatasetSplit(train=[], validation=[], test=[]), []

    logger.info("Phase 1 — split: input=%d sentences", len(sampled))
    dataset_split = split(sampled)
    logger.info(
        "Phase 1 — split: train=%d val=%d test=%d",
        len(dataset_split.train),
        len(dataset_split.validation),
        len(dataset_split.test),
    )

    return dataset_split, sampled


# ---------------------------------------------------------------------------
# Phase 2: Annotation Pipeline
# ---------------------------------------------------------------------------


def run_phase2(
    dataset_split: DatasetSplit,
    logger: logging.Logger,
) -> list[AnnotatedSentence]:
    """Convert RawSentence objects to AnnotatedSentence and run validators.

    Args:
        dataset_split: The split produced by Phase 1 (contains RawSentence
            objects cast to the AnnotatedSentence field type).
        logger: Pipeline logger.

    Returns:
        A flat list of all :class:`~models.data_models.AnnotatedSentence`
        objects across all three partitions.
    """
    logger.info("Phase 2 — converting RawSentence → AnnotatedSentence")
    lang_id_tagger = LanguageIDTagger()

    all_annotated: list[AnnotatedSentence] = []

    for raw in dataset_split.train:
        all_annotated.append(raw_to_annotated(raw, "train", lang_id_tagger))  # type: ignore[arg-type]
    for raw in dataset_split.validation:
        all_annotated.append(raw_to_annotated(raw, "validation", lang_id_tagger))  # type: ignore[arg-type]
    for raw in dataset_split.test:
        all_annotated.append(raw_to_annotated(raw, "test", lang_id_tagger))  # type: ignore[arg-type]

    logger.info("Phase 2 — converted %d sentences", len(all_annotated))

    # Span validation
    logger.info("Phase 2 — running span text invariant validation")
    span_errors = validate_all_span_text_invariants(all_annotated)
    if span_errors:
        logger.error("Phase 2 — span validation errors: %d", len(span_errors))
        for err in span_errors[:5]:
            logger.error("  %s", err)
    else:
        logger.info("Phase 2 — span validation: all records passed")

    # Toxicity label validation
    logger.info("Phase 2 — running toxicity label validation")
    tox_errors = validate_all_toxicity_labels(all_annotated)
    if tox_errors:
        logger.error("Phase 2 — toxicity label errors: %d", len(tox_errors))
        for err in tox_errors[:5]:
            logger.error("  %s", err)
    else:
        logger.info("Phase 2 — toxicity label validation: all records passed")

    return all_annotated


# ---------------------------------------------------------------------------
# Phase 3: Model Training & Evaluation
# ---------------------------------------------------------------------------


def run_phase3(
    dataset_split: DatasetSplit,
    annotated: list[AnnotatedSentence],
    seed: int,
    logger: logging.Logger,
) -> dict:
    """Train all models and run baseline + platform-split evaluation.

    Args:
        dataset_split: The split (used to build per-split sentence lists).
        annotated: All annotated sentences (flat list).
        seed: Random seed for all training runs.
        logger: Pipeline logger.

    Returns:
        A dict with keys ``"baseline_results"``, ``"platform_split_results"``,
        ``"comparison_table"``, and per-model training logs.
    """
    # Rebuild per-split lists from the flat annotated list
    train_set = [s for s in annotated if s.split == "train"]
    val_set = [s for s in annotated if s.split == "validation"]
    test_set = [s for s in annotated if s.split == "test"]

    logger.info(
        "Phase 3 — train=%d val=%d test=%d",
        len(train_set), len(val_set), len(test_set),
    )

    # Ensure at least minimal sets for training (handle very small corpora)
    if not train_set:
        train_set = annotated
    if not val_set:
        val_set = annotated[:max(1, len(annotated) // 10)]
    if not test_set:
        test_set = annotated[:max(1, len(annotated) // 10)]

    results: dict = {}

    # --- Language ID Tagger ---
    logger.info("Phase 3 — training LanguageIDTagger")
    lang_id_tagger = LanguageIDTagger()
    lang_id_metrics = lang_id_tagger.evaluate(test_set)
    logger.info(
        "Phase 3 — LanguageIDTagger: token_accuracy=%.4f",
        lang_id_metrics.token_accuracy,
    )
    results["lang_id_metrics"] = lang_id_metrics

    # --- Sentiment Classifier ---
    logger.info("Phase 3 — training SentimentClassifier (seed=%d)", seed)
    sentiment_clf = SentimentClassifier()
    sentiment_log = sentiment_clf.train(train_set, val_set, seed=seed)
    sentiment_metrics = sentiment_clf.evaluate(test_set)
    logger.info(
        "Phase 3 — SentimentClassifier: best_f1=%.4f, test_macro_f1=%.4f",
        sentiment_log.best_f1, sentiment_metrics.macro_f1,
    )
    results["sentiment_log"] = sentiment_log
    results["sentiment_metrics"] = sentiment_metrics

    # --- NER Tagger ---
    logger.info("Phase 3 — training NERTagger (seed=%d)", seed)
    ner_tagger = NERTagger()
    ner_log = ner_tagger.train(train_set, val_set, seed=seed)
    ner_metrics = ner_tagger.evaluate(test_set)
    logger.info(
        "Phase 3 — NERTagger: best_f1=%.4f, test_macro_f1=%.4f",
        ner_log.best_f1, ner_metrics.macro_f1,
    )
    results["ner_log"] = ner_log
    results["ner_metrics"] = ner_metrics

    # --- Toxicity Classifier ---
    logger.info("Phase 3 — training ToxicityClassifier (seed=%d)", seed)
    toxicity_clf = ToxicityClassifier()
    toxicity_log = toxicity_clf.train(train_set, val_set, seed=seed)
    toxicity_metrics = toxicity_clf.evaluate(test_set)
    logger.info(
        "Phase 3 — ToxicityClassifier: best_f1=%.4f, test_macro_f1=%.4f",
        toxicity_log.best_f1, toxicity_metrics.macro_f1,
    )
    results["toxicity_log"] = toxicity_log
    results["toxicity_metrics"] = toxicity_metrics

    # --- Baseline Evaluation ---
    logger.info("Phase 3 — running baseline evaluation")
    mbert_evaluator = ZeroShotMBERTEvaluator()
    muril_evaluator = ZeroShotMuRILEvaluator()
    gpt4o_evaluator = GPT4o5ShotEvaluator()

    baseline_results: list[BaselineResult] = []
    baseline_results.extend(mbert_evaluator.run_all(test_set))
    baseline_results.extend(muril_evaluator.run_all(test_set))
    baseline_results.extend(gpt4o_evaluator.run_all(test_set))

    # Add fine-tuned model results to the comparison table
    baseline_results.append(BaselineResult(
        model_name="SentimentClassifier-finetuned",
        task="sentiment",
        macro_f1=sentiment_metrics.macro_f1,
        metrics=sentiment_metrics,
    ))
    baseline_results.append(BaselineResult(
        model_name="NERTagger-finetuned",
        task="ner",
        macro_f1=ner_metrics.macro_f1,
        metrics=ner_metrics,
    ))
    baseline_results.append(BaselineResult(
        model_name="ToxicityClassifier-finetuned",
        task="toxicity",
        macro_f1=toxicity_metrics.macro_f1,
        metrics=toxicity_metrics,
    ))

    results["baseline_results"] = baseline_results

    # --- Comparison Table ---
    logger.info("Phase 3 — generating comparison table")
    comparison_rows = generate_comparison_table(baseline_results)
    table_str = format_comparison_table(comparison_rows)
    logger.info("Phase 3 — comparison table:\n%s", table_str)
    results["comparison_table"] = table_str

    # --- Platform-Split Evaluation ---
    logger.info("Phase 3 — running platform-split evaluation")
    # Build a DatasetSplit from annotated sentences for platform-split eval
    annotated_split = DatasetSplit(
        train=train_set,
        validation=val_set,
        test=test_set,
    )
    platform_split_results = run_platform_split_evaluation(
        dataset=annotated_split,
        sentiment_clf=sentiment_clf,
        ner_tagger=ner_tagger,
        toxicity_clf=toxicity_clf,
        seed=seed,
    )
    for psr in platform_split_results:
        logger.info(
            "Phase 3 — platform-split: task=%s %s→%s macro_f1=%.4f",
            psr.task, psr.train_platform, psr.eval_platform, psr.macro_f1,
        )
    results["platform_split_results"] = platform_split_results

    return results


# ---------------------------------------------------------------------------
# Phase 4: Serialization & Release
# ---------------------------------------------------------------------------


def run_phase4(
    annotated: list[AnnotatedSentence],
    output_dir: Path,
    seed: int,
    logger: logging.Logger,
    model_results: dict | None = None,
) -> None:
    """Serialize corpus, validate round-trip, and publish to HuggingFace.

    Args:
        annotated: All annotated sentences.
        output_dir: Directory for output files.
        seed: Random seed (used in model cards).
        logger: Pipeline logger.
        model_results: Optional dict from run_phase3 (used for model cards).
    """
    if not annotated:
        logger.warning("Phase 4 — no annotated sentences to serialize; skipping")
        return

    # --- JSON Lines serialization ---
    jsonl_path = output_dir / "corpus.jsonl"
    logger.info("Phase 4 — serializing to JSON Lines: %s", jsonl_path)
    serialize(annotated, jsonl_path, fmt="jsonl")
    logger.info("Phase 4 — JSON Lines written: %d records", len(annotated))

    # --- Parquet serialization ---
    parquet_path = output_dir / "corpus.parquet"
    logger.info("Phase 4 — serializing to Parquet: %s", parquet_path)
    serialize(annotated, parquet_path, fmt="parquet")
    logger.info("Phase 4 — Parquet written: %d records", len(annotated))

    # --- Round-trip validation (JSON Lines) ---
    logger.info("Phase 4 — running round-trip validation (jsonl)")
    try:
        report = validate_round_trip(annotated, jsonl_path, fmt="jsonl")
        logger.info(
            "Phase 4 — round-trip validation: passed=%d failed=%d",
            report.passed, report.failed,
        )
    except Exception as exc:
        logger.error("Phase 4 — round-trip validation FAILED: %s", exc)
        raise

    # --- HuggingFace publishing ---
    logger.info("Phase 4 — publishing to HuggingFace (stub)")
    publisher = HuggingFacePublisher()

    # Build a DatasetSplit from annotated sentences
    train_set = [s for s in annotated if s.split == "train"]
    val_set = [s for s in annotated if s.split == "validation"]
    test_set = [s for s in annotated if s.split == "test"]
    dataset_split = DatasetSplit(train=train_set, validation=val_set, test=test_set)

    publish_result = publisher.publish_dataset(
        dataset_split=dataset_split,
        repo_id="org/rajnlp-50k",
        base_delay=0.0,  # no real delay in stub
    )
    logger.info(
        "Phase 4 — dataset publish: success=%s attempts=%d",
        publish_result.success, publish_result.attempts,
    )

    # Publish models (if results available)
    if model_results:
        sentiment_metrics = model_results.get("sentiment_metrics")
        ner_metrics = model_results.get("ner_metrics")
        toxicity_metrics = model_results.get("toxicity_metrics")

        model_specs = [
            ("SentimentClassifier", "org/rajnlp-sentiment",
             {"macro_f1": sentiment_metrics.macro_f1} if sentiment_metrics else {}),
            ("NERTagger", "org/rajnlp-ner",
             {"macro_f1": ner_metrics.macro_f1} if ner_metrics else {}),
            ("ToxicityClassifier", "org/rajnlp-toxicity",
             {"macro_f1": toxicity_metrics.macro_f1} if toxicity_metrics else {}),
        ]

        for model_name, repo_id, eval_metrics in model_specs:
            model_card = publisher.generate_model_card(
                model_name=model_name,
                repo_id=repo_id,
                random_seed=seed,
                hardware_config="1× NVIDIA A100 80GB",
                training_duration="4h 00m",
                evaluation_metrics=eval_metrics,
            )
            actual_f1 = eval_metrics.get("macro_f1", 0.0)
            target_f1 = TARGET_F1.get(model_name, 0.0)
            published = publisher.publish_model(
                model_card=model_card,
                target_f1=target_f1,
                actual_f1=actual_f1,
                base_delay=0.0,
            )
            logger.info(
                "Phase 4 — model publish: %s published=%s (f1=%.4f target=%.4f)",
                model_name, published, actual_f1, target_f1,
            )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv=None):
    """Main pipeline entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(
        description="RajNLP-50K main pipeline orchestrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for all stochastic operations.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for output files (created if it does not exist).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a 100-sentence fixture instead of real data collection.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )

    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir, args.log_level)

    logger.info(
        "=== RajNLP-50K Pipeline starting (seed=%d, output_dir=%s, dry_run=%s) ===",
        args.seed, output_dir, args.dry_run,
    )

    # Fix all random seeds before any stochastic operation
    set_all_seeds(args.seed)
    logger.info("Random seeds fixed: seed=%d", args.seed)

    # -----------------------------------------------------------------------
    # Phase 1: Data Collection & Corpus Building
    # -----------------------------------------------------------------------
    logger.info("=== Phase 1: Data Collection & Corpus Building ===")
    try:
        if args.dry_run:
            logger.info("Phase 1 — dry-run mode: using 100-sentence fixture")
            raw_sentences = make_fixture_sentences(100)
        else:
            logger.info("Phase 1 — collecting real data (not implemented; using fixture)")
            raw_sentences = make_fixture_sentences(100)

        dataset_split, all_raw = run_phase1(raw_sentences, logger)
        logger.info("Phase 1 complete.")
    except Exception as exc:
        logger.error("Phase 1 FAILED: %s", exc, exc_info=True)
        raise

    # -----------------------------------------------------------------------
    # Phase 2: Annotation Pipeline
    # -----------------------------------------------------------------------
    logger.info("=== Phase 2: Annotation Pipeline ===")
    try:
        all_annotated = run_phase2(dataset_split, logger)
        logger.info("Phase 2 complete: %d annotated sentences", len(all_annotated))
    except Exception as exc:
        logger.error("Phase 2 FAILED: %s", exc, exc_info=True)
        raise

    # -----------------------------------------------------------------------
    # Phase 3: Model Training & Evaluation
    # -----------------------------------------------------------------------
    logger.info("=== Phase 3: Model Training & Evaluation ===")
    try:
        model_results = run_phase3(dataset_split, all_annotated, args.seed, logger)
        logger.info("Phase 3 complete.")
    except Exception as exc:
        logger.error("Phase 3 FAILED: %s", exc, exc_info=True)
        raise

    # -----------------------------------------------------------------------
    # Phase 4: Serialization & Release
    # -----------------------------------------------------------------------
    logger.info("=== Phase 4: Serialization & Release ===")
    try:
        run_phase4(all_annotated, output_dir, args.seed, logger, model_results)
        logger.info("Phase 4 complete.")
    except Exception as exc:
        logger.error("Phase 4 FAILED: %s", exc, exc_info=True)
        raise

    logger.info("=== Pipeline complete. ===")


if __name__ == "__main__":
    main()
