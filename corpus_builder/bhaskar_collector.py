"""
Dainik Bhaskar scraper for RajNLP-50K corpus collection.

Scrapes article text from bhaskar.com — India's largest Hindi newspaper,
with strong Rajasthan coverage. The Rajasthan edition contains significant
Rajasthani-Hindi code-switching, especially in local city news.

Sections scraped:
  - /rajasthan (state-level news)
  - /local/rajasthan/jaipur, /jodhpur, /udaipur, etc.

Usage:
    from corpus_builder.bhaskar_collector import BhaskarCollector

    collector = BhaskarCollector()
    sentences = collector.collect(max_articles=500)

Requirements: 2.1, 2.4, 2.5
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Optional

from models.data_models import RawSentence

logger = logging.getLogger(__name__)

BASE_URL = "https://www.bhaskar.com"

# Rajasthan city sections on bhaskar.com
RAJASTHAN_SECTIONS = [
    "/rajasthan",
    "/local/rajasthan/jaipur",
    "/local/rajasthan/jodhpur",
    "/local/rajasthan/udaipur",
    "/local/rajasthan/kota",
    "/local/rajasthan/ajmer",
    "/local/rajasthan/bikaner",
    "/local/rajasthan/alwar",
    "/local/rajasthan/bharatpur",
    "/local/rajasthan/sikar",
    "/local/rajasthan/churu",
    "/local/rajasthan/nagaur",
    "/local/rajasthan/barmer",
    "/local/rajasthan/jaisalmer",
    "/local/rajasthan/pali",
    "/local/rajasthan/bhilwara",
    "/local/rajasthan/sriganganagar",
    "/local/rajasthan/hanumangarh",
    "/local/rajasthan/jhunjhunu",
    "/local/rajasthan/dausa",
]

MIN_SENTENCE_CHARS = 20
REQUEST_DELAY = 1.5
REQUEST_TIMEOUT = 15
_WHITESPACE_NORM = re.compile(r'\s+')
_SENTENCE_SPLIT = re.compile(r'(?<=[।॥?!])\s*|(?<=[.?!])\s+(?=[A-Z\u0900-\u097F])')


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _split_sentences(text: str) -> list[str]:
    text = _nfc(text)
    text = _WHITESPACE_NORM.sub(" ", text).strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text)
    result = []
    for part in parts:
        for sub in re.split(r'(?<=।)', part):
            sub = sub.strip()
            if (len(sub) >= MIN_SENTENCE_CHARS
                    and not re.match(r'^[\d\s\W]+$', sub)
                    and not sub.startswith("http")
                    and "कॉपी लिंक" not in sub
                    and "शेयर" not in sub[:10]):
                result.append(sub)
    return result


class BhaskarCollector:
    """Scraper for Dainik Bhaskar (bhaskar.com) Rajasthan edition."""

    def __init__(
        self,
        request_delay: float = REQUEST_DELAY,
        timeout: int = REQUEST_TIMEOUT,
        max_retries: int = 2,
    ) -> None:
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "hi-IN,hi;q=0.9,en;q=0.8",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Referer": "https://www.bhaskar.com/",
            })
        return self._session

    def _get_page(self, url: str) -> Optional[str]:
        session = self._get_session()
        for attempt in range(self.max_retries + 1):
            try:
                response = session.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    return response.text
                elif response.status_code in (404, 410):
                    return None
                elif response.status_code == 429:
                    wait = 60 * (attempt + 1)
                    logger.warning("Rate limited — waiting %ds", wait)
                    time.sleep(wait)
                else:
                    logger.debug("HTTP %d for %s", response.status_code, url)
                    if attempt < self.max_retries:
                        time.sleep(self.request_delay * 2)
            except Exception as exc:
                logger.debug("Request failed (attempt %d): %s — %s", attempt + 1, url, exc)
                if attempt < self.max_retries:
                    time.sleep(self.request_delay * 2)
        return None

    def _extract_article_urls(self, html: str) -> list[str]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("pip install beautifulsoup4 lxml")

        soup = BeautifulSoup(html, "lxml")
        urls = set()

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            # Bhaskar article URLs: /news/... or /local/rajasthan/.../news/...
            if "/news/" in href and re.search(r'-\d{6,}', href):
                if href.startswith("/"):
                    href = BASE_URL + href
                if href.startswith(BASE_URL):
                    urls.add(href.split("?")[0])  # strip query params

        return list(urls)

    def _extract_sentences(self, html: str, url: str) -> list[str]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("pip install beautifulsoup4 lxml")

        soup = BeautifulSoup(html, "lxml")
        raw_texts: list[str] = []

        # Bhaskar article content selectors
        content_selectors = [
            "div.art-content",
            "div[class*='article-content']",
            "div[class*='story-content']",
            "div.db-article",
            "div[class*='content-area']",
            "article",
            "div.news-detail",
            "div[class*='detail']",
        ]

        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                for tag in content.find_all(["p", "h2", "h3", "li"]):
                    text = tag.get_text(separator=" ", strip=True)
                    if text and len(text) > MIN_SENTENCE_CHARS:
                        raw_texts.append(text)
                if raw_texts:
                    break

        # Fallback: all paragraphs with Devanagari
        if not raw_texts:
            for p in soup.find_all("p"):
                text = p.get_text(separator=" ", strip=True)
                if (len(text) >= MIN_SENTENCE_CHARS
                        and any('\u0900' <= c <= '\u097F' for c in text)):
                    parent_classes = " ".join(
                        p.parent.get("class", []) if p.parent else []
                    ).lower()
                    if not any(skip in parent_classes for skip in
                               ["nav", "menu", "header", "footer", "ad", "banner"]):
                        raw_texts.append(text)

        sentences = []
        for text in raw_texts:
            sentences.extend(_split_sentences(text))
        return sentences

    def collect_section(
        self,
        section_path: str,
        max_articles: int = 100,
    ) -> list[RawSentence]:
        url = BASE_URL + section_path
        logger.info("Collecting from: %s", url)

        html = self._get_page(url)
        if html is None:
            logger.warning("Could not fetch: %s", url)
            return []

        article_urls = self._extract_article_urls(html)
        logger.info("Found %d articles in %s", len(article_urls), section_path)

        sentences: list[RawSentence] = []
        for i, art_url in enumerate(article_urls[:max_articles]):
            art_html = self._get_page(art_url)
            if art_html is None:
                continue

            art_sentences = self._extract_sentences(art_html, art_url)
            for text in art_sentences:
                sentences.append(RawSentence(
                    text=text,
                    source_url=art_url,
                    collected_at=datetime.now(tz=timezone.utc),
                    platform="sharechat",  # non-Twitter source
                    sentence_id=str(uuid.uuid4()),
                ))

            time.sleep(self.request_delay)

        logger.info("Section %s: %d sentences from %d articles",
                    section_path, len(sentences), min(len(article_urls), max_articles))
        return sentences

    def collect(
        self,
        sections: list[str] | None = None,
        max_articles_per_section: int = 50,
    ) -> list[RawSentence]:
        if sections is None:
            sections = RAJASTHAN_SECTIONS

        all_sentences: list[RawSentence] = []
        for section in sections:
            try:
                s = self.collect_section(section, max_articles=max_articles_per_section)
                all_sentences.extend(s)
                logger.info("Running total: %d sentences", len(all_sentences))
            except Exception as exc:
                logger.error("Error in section %s: %s", section, exc)
                continue

        logger.info("Bhaskar collection complete: %d total sentences", len(all_sentences))
        return all_sentences


def main(argv=None):
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Collect sentences from Dainik Bhaskar Rajasthan (bhaskar.com).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sections", nargs="+", default=["/rajasthan", "/local/rajasthan/jaipur",
                        "/local/rajasthan/jodhpur", "/local/rajasthan/udaipur"])
    parser.add_argument("--max-articles", type=int, default=50)
    parser.add_argument("--output", default="data/bhaskar_raw.jsonl")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    collector = BhaskarCollector(request_delay=args.delay)
    sentences = collector.collect(
        sections=args.sections,
        max_articles_per_section=args.max_articles,
    )

    if not sentences:
        print("No sentences collected.")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for s in sentences:
            fh.write(json.dumps({
                "text": s.text, "source_url": s.source_url,
                "collected_at": s.collected_at.isoformat(),
                "platform": s.platform, "sentence_id": s.sentence_id,
            }, ensure_ascii=False) + "\n")

    print(f"\n✓ Collected {len(sentences)} sentences → {output_path}")
    print(f"\nNext: python run_pipeline.py --manual-data {output_path},data/patrika_full.jsonl,data/patrika_large.jsonl --seed 42 --output-dir output/run_combined")


if __name__ == "__main__":
    main()
