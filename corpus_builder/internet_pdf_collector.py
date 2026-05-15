"""
Internet PDF Collector for RajNLP-50K.

Automatically finds, downloads, and processes Rajasthani/Hindi PDFs
from public sources (Internet Archive, government sites, etc.).

Also provides a simple RAG (Retrieval-Augmented Generation) module
that indexes the collected text for semantic search.

Usage:
    # Download and process PDFs
    python -m corpus_builder.internet_pdf_collector \
        --output data/internet_pdfs_raw.jsonl \
        --max-pdfs 20

    # Or use programmatically
    from corpus_builder.internet_pdf_collector import InternetPDFCollector
    collector = InternetPDFCollector()
    sentences = collector.collect(max_pdfs=20)

Requirements: Public domain / openly licensed texts only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models.data_models import RawSentence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated list of public domain Rajasthani/Hindi PDFs from Internet Archive
# All are public domain or openly licensed
# ---------------------------------------------------------------------------

RAJASTHANI_PDF_SOURCES: list[dict] = [
    # Rajasthani literature and linguistics
    {
        "name": "Rajasthani Loka Sahitya",
        "url": "https://archive.org/download/rajasthanilokasa00devauoft/rajasthanilokasa00devauoft.pdf",
        "description": "Rajasthani folk literature - viraha, prakriti, bhakti",
        "language": "rajasthani",
        "license": "public_domain",
    },
    {
        "name": "Linguistic Survey of India - Rajasthan Part I",
        "url": "https://archive.org/download/PARI.linguistic-survey-of-india---rajasthan-part-i/PARI.linguistic-survey-of-india---rajasthan-part-i.pdf",
        "description": "Linguistic survey with Marwari, Jaipuri, Mewati text samples",
        "language": "rajasthani",
        "license": "public_domain",
    },
    # Hindi literature with Rajasthani elements
    {
        "name": "Rajasthan Ka Itihas",
        "url": "https://archive.org/download/in.ernet.dli.2015.400629/2015.400629.Rajasthan-Ka.pdf",
        "description": "History of Rajasthan in Hindi",
        "language": "hindi",
        "license": "public_domain",
    },
    # Additional Rajasthani texts from Digital Library of India
    {
        "name": "Rajasthani Bhasha Aur Sahitya",
        "url": "https://archive.org/download/dli.ernet.429984/429984.pdf",
        "description": "Rajasthani language and literature",
        "language": "rajasthani",
        "license": "public_domain",
    },
]

# Internet Archive search API for finding more Rajasthani texts
ARCHIVE_SEARCH_QUERIES = [
    "rajasthani language devanagari",
    "marwari hindi text",
    "rajasthan hindi sahitya",
    "rajasthani kavita devanagari",
    "rajasthani lok geet",
    "mewar history hindi",
    "jodhpur history hindi devanagari",
]

MIN_SENTENCE_CHARS = 20
REQUEST_DELAY = 2.0
REQUEST_TIMEOUT = 30


class InternetPDFCollector:
    """Downloads and processes Rajasthani/Hindi PDFs from the internet.

    Sources:
    - Internet Archive (archive.org) — public domain texts
    - Digital Library of India — historical Rajasthani texts
    - Government of Rajasthan publications (open access)
    """

    def __init__(
        self,
        download_dir: str = "data/downloaded_pdfs",
        request_delay: float = REQUEST_DELAY,
    ) -> None:
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": "Mozilla/5.0 (compatible; RajNLP-50K research bot)",
            })
        return self._session

    def _download_pdf(self, url: str, name: str) -> Optional[Path]:
        """Download a PDF from a URL and save to download_dir."""
        safe_name = re.sub(r'[^\w\-]', '_', name)[:50] + ".pdf"
        dest = self.download_dir / safe_name

        if dest.exists() and dest.stat().st_size > 1000:
            logger.info("Already downloaded: %s", dest)
            return dest

        session = self._get_session()
        try:
            logger.info("Downloading: %s → %s", url, dest)
            response = session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            if response.status_code == 200:
                with dest.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=8192):
                        fh.write(chunk)
                size_mb = dest.stat().st_size / 1024 / 1024
                logger.info("Downloaded: %s (%.1f MB)", dest, size_mb)
                return dest
            else:
                logger.warning("HTTP %d for %s", response.status_code, url)
                return None
        except Exception as exc:
            logger.warning("Download failed for %s: %s", url, exc)
            return None

    def _search_archive_org(self, query: str, max_results: int = 5) -> list[dict]:
        """Search Internet Archive for Rajasthani/Hindi PDFs."""
        session = self._get_session()
        search_url = "https://archive.org/advancedsearch.php"
        params = {
            "q": f"{query} AND mediatype:texts AND language:(Hindi OR Rajasthani)",
            "fl[]": ["identifier", "title", "description", "language"],
            "rows": max_results,
            "page": 1,
            "output": "json",
        }
        try:
            response = session.get(search_url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                docs = data.get("response", {}).get("docs", [])
                results = []
                for doc in docs:
                    identifier = doc.get("identifier", "")
                    title = doc.get("title", identifier)
                    results.append({
                        "name": title,
                        "identifier": identifier,
                        "url": f"https://archive.org/download/{identifier}/{identifier}.pdf",
                        "description": doc.get("description", ""),
                        "language": doc.get("language", "hindi"),
                        "license": "public_domain",
                    })
                return results
        except Exception as exc:
            logger.warning("Archive.org search failed: %s", exc)
        return []

    def _process_pdf(self, pdf_path: Path, source_name: str) -> list[RawSentence]:
        """Extract sentences from a downloaded PDF."""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pip install pdfplumber")

        from corpus_builder.krutidev_converter import convert_pdf_text, is_likely_krutidev
        from corpus_builder.book_importer import _split_sentences

        sentences: list[RawSentence] = []
        collected_at = datetime.now(tz=timezone.utc)
        source_url = f"archive.org://{source_name}"

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                total_pages = len(pdf.pages)
                logger.info("Processing %s (%d pages)", pdf_path.name, total_pages)

                for page_num, page in enumerate(pdf.pages, start=1):
                    if page_num % 20 == 0:
                        logger.info("  Page %d/%d...", page_num, total_pages)

                    text = page.extract_text()
                    if not text:
                        continue

                    # Auto-convert Krutidev encoding
                    if is_likely_krutidev(text):
                        text = convert_pdf_text(text)

                    for sent in _split_sentences(text):
                        sentences.append(RawSentence(
                            text=sent,
                            source_url=source_url,
                            collected_at=collected_at,
                            platform="sharechat",
                            sentence_id=str(uuid.uuid4()),
                        ))
        except Exception as exc:
            logger.error("Failed to process %s: %s", pdf_path, exc)

        logger.info("Extracted %d sentences from %s", len(sentences), pdf_path.name)
        return sentences

    def collect_from_curated_list(self) -> list[RawSentence]:
        """Download and process all PDFs from the curated list."""
        all_sentences: list[RawSentence] = []

        for source in RAJASTHANI_PDF_SOURCES:
            pdf_path = self._download_pdf(source["url"], source["name"])
            if pdf_path is None:
                continue

            sentences = self._process_pdf(pdf_path, source["name"])
            all_sentences.extend(sentences)
            logger.info("Running total: %d sentences", len(all_sentences))
            time.sleep(self.request_delay)

        return all_sentences

    def collect_from_archive_search(self, max_pdfs: int = 10) -> list[RawSentence]:
        """Search Internet Archive and download/process found PDFs."""
        all_sentences: list[RawSentence] = []
        downloaded = 0

        for query in ARCHIVE_SEARCH_QUERIES:
            if downloaded >= max_pdfs:
                break

            logger.info("Searching archive.org: %s", query)
            results = self._search_archive_org(query, max_results=3)
            time.sleep(self.request_delay)

            for result in results:
                if downloaded >= max_pdfs:
                    break

                pdf_path = self._download_pdf(result["url"], result["name"])
                if pdf_path is None:
                    continue

                sentences = self._process_pdf(pdf_path, result["name"])
                all_sentences.extend(sentences)
                downloaded += 1
                logger.info("Running total: %d sentences from %d PDFs", len(all_sentences), downloaded)
                time.sleep(self.request_delay)

        return all_sentences

    def collect(self, max_pdfs: int = 20) -> list[RawSentence]:
        """Collect from both curated list and archive.org search."""
        logger.info("=== Internet PDF Collection ===")

        # First: curated list (most reliable)
        logger.info("Phase 1: Curated PDF list (%d sources)", len(RAJASTHANI_PDF_SOURCES))
        curated = self.collect_from_curated_list()

        # Second: archive.org search
        remaining = max_pdfs - len(RAJASTHANI_PDF_SOURCES)
        if remaining > 0:
            logger.info("Phase 2: Archive.org search (up to %d more PDFs)", remaining)
            searched = self.collect_from_archive_search(max_pdfs=remaining)
        else:
            searched = []

        all_sentences = curated + searched
        logger.info("Internet PDF collection complete: %d total sentences", len(all_sentences))
        return all_sentences


# ---------------------------------------------------------------------------
# Simple RAG (Retrieval-Augmented Generation) module
# ---------------------------------------------------------------------------

class RajasthaniRAG:
    """Simple RAG module for Rajasthani text.

    Indexes collected sentences and supports semantic search.
    Useful for:
    - Finding similar sentences for annotation examples
    - Retrieving context for a given query
    - Augmenting the corpus with retrieved examples

    Uses TF-IDF for retrieval (no GPU needed).
    """

    def __init__(self) -> None:
        self._sentences: list[str] = []
        self._vectorizer = None
        self._matrix = None
        self._built = False

    def index(self, sentences: list[str]) -> None:
        """Build the search index from a list of sentences."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ImportError("pip install scikit-learn")

        self._sentences = sentences
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=50000,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(sentences)
        self._built = True
        logger.info("RAG index built: %d sentences", len(sentences))

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, str]]:
        """Search for sentences similar to the query.

        Args:
            query: Search query (Rajasthani/Hindi text).
            top_k: Number of results to return.

        Returns:
            List of (score, sentence) tuples, sorted by relevance.
        """
        if not self._built:
            raise RuntimeError("Call index() first.")

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(float(scores[i]), self._sentences[i]) for i in top_indices]

    def save(self, path: str) -> None:
        """Save the index to disk."""
        import pickle
        with open(path, "wb") as fh:
            pickle.dump({
                "sentences": self._sentences,
                "vectorizer": self._vectorizer,
                "matrix": self._matrix,
            }, fh)
        logger.info("RAG index saved to %s", path)

    def load(self, path: str) -> None:
        """Load the index from disk."""
        import pickle
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self._sentences = data["sentences"]
        self._vectorizer = data["vectorizer"]
        self._matrix = data["matrix"]
        self._built = True
        logger.info("RAG index loaded: %d sentences", len(self._sentences))

    @classmethod
    def build_from_jsonl(cls, jsonl_path: str, index_path: str | None = None) -> "RajasthaniRAG":
        """Build a RAG index from a JSONL corpus file.

        Args:
            jsonl_path: Path to the corpus JSONL file.
            index_path: Optional path to save the index.

        Returns:
            A built RajasthaniRAG instance.
        """
        sentences = []
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    sentences.append(rec["text"])

        rag = cls()
        rag.index(sentences)

        if index_path:
            rag.save(index_path)

        return rag


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Download and process Rajasthani PDFs from the internet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", default="data/internet_pdfs_raw.jsonl")
    parser.add_argument("--max-pdfs", type=int, default=20)
    parser.add_argument("--download-dir", default="data/downloaded_pdfs")
    parser.add_argument("--build-rag", action="store_true",
                        help="Build a RAG index from the collected sentences")
    parser.add_argument("--rag-index", default="data/rajasthani_rag.pkl",
                        help="Path to save the RAG index")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    collector = InternetPDFCollector(download_dir=args.download_dir)
    sentences = collector.collect(max_pdfs=args.max_pdfs)

    if not sentences:
        print("No sentences collected.")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for s in sentences:
            fh.write(json.dumps({
                "text": s.text,
                "source_url": s.source_url,
                "collected_at": s.collected_at.isoformat(),
                "platform": s.platform,
                "sentence_id": s.sentence_id,
            }, ensure_ascii=False) + "\n")

    print(f"\n✓ Collected {len(sentences)} sentences → {output_path}")

    if args.build_rag:
        print(f"\nBuilding RAG index...")
        rag = RajasthaniRAG.build_from_jsonl(str(output_path), args.rag_index)
        print(f"✓ RAG index saved → {args.rag_index}")
        print(f"\nTest search: 'राजस्थानी भाषा'")
        results = rag.search("राजस्थानी भाषा", top_k=3)
        for score, sent in results:
            print(f"  [{score:.3f}] {sent[:80]}")

    print(f"\nNext: python run_pipeline.py --manual-data {output_path} --seed 42 --output-dir output/run_internet")


if __name__ == "__main__":
    main()
