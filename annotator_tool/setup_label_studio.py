#!/usr/bin/env python3
"""
Label Studio setup script for RajNLP-50K.

This script:
1. Checks if Label Studio is installed (installs if not)
2. Creates all 3 annotation projects (sentiment, NER, toxicity)
3. Imports all 50K sentences from the corpus split file
4. Prints the URL to open in a browser

Usage:
    python -m annotator_tool.setup_label_studio [--corpus output/corpus_build/corpus_raw_split.jsonl]

Requirements: 4.1, 4.2, 4.3, 4.4, 16.2, 16.3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_STUDIO_PORT = 8080
LABEL_STUDIO_URL = f"http://localhost:{LABEL_STUDIO_PORT}"
LABEL_STUDIO_TOKEN_FILE = Path(".label_studio_token")

CONFIGS_DIR = Path(__file__).parent / "label_studio_configs"

PROJECT_CONFIGS = {
    "rajnlp-sentiment": {
        "title": "RajNLP-50K Sentiment Annotation",
        "description": "3-class sentiment labeling (positive / neutral / negative)",
        "config_file": CONFIGS_DIR / "rajnlp-sentiment.xml",
        "color": "#FF6B6B",
    },
    "rajnlp-ner": {
        "title": "RajNLP-50K NER Annotation",
        "description": "Span-level NER labeling (PER / LOC / ORG)",
        "config_file": CONFIGS_DIR / "rajnlp-ner.xml",
        "color": "#4ECDC4",
    },
    "rajnlp-toxicity": {
        "title": "RajNLP-50K Toxicity Annotation",
        "description": "Multi-label toxicity detection (caste_slur / religious / gender / general)",
        "config_file": CONFIGS_DIR / "rajnlp-toxicity.xml",
        "color": "#45B7D1",
    },
}


# ---------------------------------------------------------------------------
# Installation check
# ---------------------------------------------------------------------------

def ensure_label_studio_installed() -> bool:
    """Check if label-studio is installed; install if not."""
    try:
        result = subprocess.run(
            ["label-studio", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            logger.info("Label Studio already installed: %s", result.stdout.strip())
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    logger.info("Label Studio not found. Installing...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "label-studio==1.11.0"],
        capture_output=False
    )
    if result.returncode != 0:
        logger.error("Failed to install Label Studio. Run manually: pip install label-studio")
        return False

    logger.info("Label Studio installed successfully.")
    return True


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get_headers(token: str) -> dict:
    return {"Authorization": f"Token {token}", "Content-Type": "application/json"}


def _api(method: str, path: str, token: str, data: dict | None = None) -> dict | list | None:
    """Make a Label Studio API call using urllib (no requests dependency)."""
    import urllib.request
    import urllib.error

    url = f"{LABEL_STUDIO_URL}/api/{path}"
    headers = _get_headers(token)
    body = json.dumps(data).encode("utf-8") if data else None

    req = urllib.request.Request(url, data=body, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        logger.error("API %s %s failed: %s %s", method, path, e.code, e.read().decode())
        return None
    except Exception as e:
        logger.error("API %s %s error: %s", method, path, e)
        return None


# ---------------------------------------------------------------------------
# Project creation
# ---------------------------------------------------------------------------

def create_project(token: str, name: str, config: dict) -> int | None:
    """Create a Label Studio project. Returns project ID."""
    label_config = config["config_file"].read_text(encoding="utf-8")

    payload = {
        "title": config["title"],
        "description": config["description"],
        "label_config": label_config,
        "maximum_annotations": 3,
        "overlap_cohort_percentage": 100,
        "show_annotation_history": False,
        "color": config["color"],
    }

    result = _api("POST", "projects/", token, payload)
    if result and "id" in result:
        project_id = result["id"]
        logger.info("Created project '%s' (id=%d)", config["title"], project_id)
        return project_id
    else:
        logger.error("Failed to create project '%s'", name)
        return None


# ---------------------------------------------------------------------------
# Task import
# ---------------------------------------------------------------------------

def import_tasks(token: str, project_id: int, corpus_path: Path, max_tasks: int | None = None) -> int:
    """Import sentences from corpus JSONL as Label Studio tasks."""
    import urllib.request

    tasks = []
    with corpus_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tasks.append({
                "data": {
                    "text": obj["text"],
                    "sentence_id": obj["sentence_id"],
                    "platform": obj["platform"],
                    "split": obj["split"],
                    "source_url": obj.get("source_url", ""),
                }
            })
            if max_tasks and len(tasks) >= max_tasks:
                break

    logger.info("Importing %d tasks into project %d...", len(tasks), project_id)

    # Import in batches of 1000
    batch_size = 1000
    imported = 0
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        result = _api("POST", f"projects/{project_id}/import", token, batch)
        if result:
            imported += len(batch)
            logger.info("  Imported batch %d/%d (%d tasks)", i // batch_size + 1,
                        (len(tasks) + batch_size - 1) // batch_size, len(batch))
        else:
            logger.error("  Failed to import batch starting at index %d", i)

    return imported


# ---------------------------------------------------------------------------
# Main setup
# ---------------------------------------------------------------------------

def setup(corpus_path: Path, token: str, max_tasks_per_project: int | None = None) -> dict:
    """Create all 3 projects and import tasks.

    Args:
        corpus_path: Path to corpus_raw_split.jsonl
        token: Label Studio API token
        max_tasks_per_project: Limit tasks per project (None = all 50K)

    Returns:
        Dict mapping project name → project_id
    """
    project_ids = {}

    for name, config in PROJECT_CONFIGS.items():
        logger.info("=== Setting up project: %s ===", name)
        project_id = create_project(token, name, config)
        if project_id is None:
            logger.error("Skipping task import for %s", name)
            continue

        imported = import_tasks(token, project_id, corpus_path, max_tasks=max_tasks_per_project)
        logger.info("Project '%s': %d tasks imported (id=%d)", name, imported, project_id)
        project_ids[name] = project_id

    return project_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Set up Label Studio for RajNLP-50K annotation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        default="output/corpus_build/corpus_raw_split.jsonl",
        help="Path to the corpus JSONL file.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Label Studio API token. If not provided, reads from .label_studio_token file.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Max tasks to import per project (default: all). Use a small number for testing.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check if Label Studio is installed and running; don't create projects.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-8s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        logger.error("Corpus file not found: %s", corpus_path)
        logger.error("Run first: python -m corpus_builder.build_corpus --skip-minhash")
        return 1

    # Get token
    token = args.token
    if not token and LABEL_STUDIO_TOKEN_FILE.exists():
        token = LABEL_STUDIO_TOKEN_FILE.read_text().strip()
    if not token:
        token = os.environ.get("LABEL_STUDIO_TOKEN", "")

    print("\n" + "="*60)
    print("  RajNLP-50K Label Studio Setup")
    print("="*60)
    print()
    print("STEP 1: Start Label Studio")
    print("  Run this in a separate terminal:")
    print()
    print("    label-studio start --port 8080")
    print()
    print("  Then open: http://localhost:8080")
    print("  Create an account, then get your API token from:")
    print("  Account & Settings → Access Token")
    print()
    print("STEP 2: Import projects and tasks")
    print("  Run this script with your token:")
    print()
    print("    python -m annotator_tool.setup_label_studio --token YOUR_TOKEN_HERE")
    print()

    if args.check_only:
        logger.info("--check-only mode: skipping project creation")
        return 0

    if not token:
        print("No token provided. Please follow STEP 1 above first.")
        print("Then re-run with: --token YOUR_TOKEN_HERE")
        return 0

    # Test connection
    logger.info("Testing connection to Label Studio at %s...", LABEL_STUDIO_URL)
    result = _api("GET", "projects/", token)
    if result is None:
        logger.error("Cannot connect to Label Studio at %s", LABEL_STUDIO_URL)
        logger.error("Make sure Label Studio is running: label-studio start --port 8080")
        return 1

    logger.info("Connected to Label Studio successfully.")

    # Create projects and import tasks
    project_ids = setup(
        corpus_path=corpus_path,
        token=token,
        max_tasks_per_project=args.max_tasks,
    )

    print()
    print("="*60)
    print("  Setup Complete!")
    print("="*60)
    print()
    print(f"  Created {len(project_ids)} projects:")
    for name, pid in project_ids.items():
        print(f"    {name}: {LABEL_STUDIO_URL}/projects/{pid}/")
    print()
    print("  Next steps:")
    print("  1. Open Label Studio: http://localhost:8080")
    print("  2. Invite annotators (Settings → Members)")
    print("  3. Annotators start labeling sentences")
    print("  4. Export annotations when done:")
    print("     python -m annotator_tool.export_converter --project-id <ID> --token <TOKEN>")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
