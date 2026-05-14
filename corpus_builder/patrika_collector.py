"""
Rajasthan Patrika scraper for RajNLP-50K corpus collection.

Scrapes article text and reader comments from patrika.com — one of the
largest Hindi-Rajasthani newspapers. The comment sections are particularly
rich in Rajasthani-Hindi code-switching.

Sources scraped:
  - Article body text (Hindi with Rajasthani dialect words)
  - Reader comments (highest code-switching density)
  - Section pages: rajasthan-news, jaipur-news, jodhpur-news, etc.

Usage:
    from corpus_builder.patrika_collector import PatrikaCollector

    collector = PatrikaCollector()
    sentences = collector.collect(max_articles=500)

    # Or target specific sections
    sentences = collector.collect_section("rajasthan-news", max_articles=200)

Requirements: 2.1, 2.4, 2.5 (same as ShareChat collector)
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://www.patrika.com"

# Sections with high Rajasthani dialect content
RAJASTHAN_SECTIONS = [
    "rajasthan-news",
    "jaipur-news",
    "jodhpur-news",
    "udaipur-news",
    "kota-news",
    "ajmer-news",
    "bikaner-news",
    "alwar-news",
    "bharatpur-news",
    "sikar-news",
    "churu-news",
    "nagaur-news",
    "barmer-news",
    "jaisalmer-news",
    "pali-news",
    "rajsamand-news",
]

# Minimum sentence length in characters
MIN_SENTENCE_CHARS = 20

# Delay between requests (seconds) — be respectful to the server
REQUEST_DELAY = 1.5

# Default timeout for requests
REQUEST_TIMEOUT = 15


class PatrikaCollector:
    """Scraper for Rajasthan Patrika (patrika.com).

    Collects article text and reader comments which contain
    Rajasthani-Hindi code-switched language.

    Requirements: 2.1, 2.4, 2.5
    """

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
        """Get or create a requests session with appropriate headers."""
        if self._session is None:
            try:
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
                })
            except ImportError:
                raise ImportError("requests is required: pip install requests")
        return self._session

    def _get_page(self, url: str) -> Optional[str]:
        """Fetch a page with retry logic.

        Args:
            url: URL to fetch.

        Returns:
            HTML content as string, or None on failure.
        """
        session = self._get_session()
        for attempt in range(self.max_retries + 1):
            try:
                response = session.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    return response.text
                elif response.status_code in (404, 410):
                    logger.debug("Page not found (HTTP %d): %s", response.status_code, url)
                    return None
                elif response.status_code == 429:
                    wait = 30 * (attempt + 1)
                    logger.warning("Rate limited (HTTP 429) — waiting %ds: %s", wait, url)
                    time.sleep(wait)
                else:
                    logger.warning("HTTP %d for %s", response.status_code, url)
                    if attempt < self.max_retries:
                        time.sleep(self.request_delay * 2)
            except Exception as exc:
                logger.warning("Request failed (attempt %d/%d): %s — %s",
                               attempt + 1, self.max_retries + 1, url, exc)
                if attempt < self.max_retries:
                    time.sleep(self.request_delay * 2)
        return None

    def _extract_article_urls(self, section_html: str, section: str) -> list[str]:
        """Extract article URLs from a section page.

        Args:
            section_html: HTML content of the section page.
            section: Section name (for logging).

        Returns:
            List of article URLs.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4 lxml")

        soup = BeautifulSoup(section_html, "lxml")
        urls = set()

        # patrika.com article links typically contain the section name and end with a number
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            # Article URLs look like: /jaipur-news/some-headline-12345678/
            if re.search(r"/\d{6,}/?$", href) or re.search(r"-\d{6,}/?$", href):
                if href.startswith("/"):
                    href = BASE_URL + href
                if href.startswith(BASE_URL):
                    urls.add(href)

        logger.debug("Found %d article URLs in section '%s'", len(urls), section)
        return list(urls)

    def _extract_sentences_from_article(self, html: str, url: str) -> list[str]:
        """Extract sentences from an article page.

        Extracts both article body text and reader comments.

        Args:
            html: HTML content of the article page.
            url: Article URL (for logging).

        Returns:
            List of sentence strings.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4 lxml")

        soup = BeautifulSoup(html, "lxml")
        raw_texts: list[str] = []

        # --- Article body ---
        # patrika.com uses various content div classes
        content_selectors = [
            "div.content-area",
            "div.article-content",
            "div.story-content",
            "div[class*='content']",
            "article",
            "div.news-content",
            "div.article-body",
        ]
        for selector in content_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                # Get all paragraph text
                for p in content_div.find_all(["p", "h2", "h3"]):
                    text = p.get_text(separator=" ", strip=True)
                    if text:
                        raw_texts.append(text)
                break

        # Fallback: get all paragraphs if no content div found
        if not raw_texts:
            for p in soup.find_all("p"):
                text = p.get_text(separator=" ", strip=True)
                if len(text) > MIN_SENTENCE_CHARS:
                    raw_texts.append(text)

        # --- Reader comments ---
        comment_selectors = [
            "div.comment-text",
            "div.comment-body",
            "div[class*='comment']",
            "span.comment-content",
            "p.comment",
        ]
        for selector in comment_selectors:
            for comment_div in soup.select(selector):
                text = comment_div.get_text(separator=" ", strip=True)
                if text:
                    raw_texts.append(text)

        # Split long texts into sentences
        sentences = []
        for text in raw_texts:
            sentences.extend(_split_into_sentences(text))

        return sentences

    def collect_section(
        self,
        section: str,
        max_articles: int = 100,
    ) -> list[RawSentence]:
        """Collect sentences from a specific Patrika section.

        Args:
            section: Section name (e.g., "rajasthan-news", "jaipur-news").
            max_articles: Maximum number of articles to scrape.

        Returns:
            List of RawSentence objects.
        """
        section_url = f"{BASE_URL}/{section}/"
        logger.info("Collecting from section: %s", section_url)

        html = self._get_page(section_url)
        if html is None:
            logger.warning("Could not fetch section page: %s", section_url)
            return []

        article_urls = self._extract_article_urls(html, section)
        logger.info("Found %d articles in section '%s'", len(article_urls), section)

        sentences: list[RawSentence] = []
        for i, url in enumerate(article_urls[:max_articles]):
            logger.debug("Scraping article %d/%d: %s", i + 1, min(len(article_urls), max_articles), url)

            article_html = self._get_page(url)
            if article_html is None:
                logger.debug("Skipping article (fetch failed): %s", url)
                continue

            article_sentences = self._extract_sentences_from_article(article_html, url)
            for text in article_sentences:
                sentences.append(RawSentence(
                    text=text,
                    source_url=url,
                    collected_at=datetime.now(tz=timezone.utc),
                    platform="sharechat",  # mapped to sharechat as non-Twitter source
                    sentence_id=str(uuid.uuid4()),
                ))

            logger.debug("Article %s: extracted %d sentences", url, len(article_sentences))
            time.sleep(self.request_delay)

        logger.info(
            "Section '%s': collected %d sentences from %d articles",
            section, len(sentences), min(len(article_urls), max_articles),
        )
        return sentences

    def collect(
        self,
        sections: list[str] | None = None,
        max_articles_per_section: int = 50,
    ) -> list[RawSentence]:
        """Collect sentences from multiple Patrika sections.

        Args:
            sections: List of section names to scrape. Defaults to all
                Rajasthan sections.
            max_articles_per_section: Maximum articles per section.

        Returns:
            Combined list of RawSentence objects from all sections.
        """
        if sections is None:
            sections = RAJASTHAN_SECTIONS

        all_sentences: list[RawSentence] = []
        for section in sections:
            try:
                section_sentences = self.collect_section(
                    section, max_articles=max_articles_per_section
                )
                all_sentences.extend(section_sentences)
                logger.info(
                    "Running total: %d sentences (after section '%s')",
                    len(all_sentences), section,
                )
            except Exception as exc:
                logger.error("Error collecting section '%s': %s", section, exc)
                continue

        logger.info("Patrika collection complete: %d total sentences", len(all_sentences))
        return all_sentences


# ---------------------------------------------------------------------------
# Sentence splitting utilities
# ---------------------------------------------------------------------------

# Sentence boundary patterns for Hindi/Devanagari text
_SENTENCE_ENDINGS = re.compile(
    r'(?<=[।?!.।\n])\s+'  # Devanagari danda (।) and standard punctuation
    r'|(?<=[?!.])\s+(?=[A-Z\u0900-\u097F])'  # Before capital/Devanagari
)

_WHITESPACE_NORM = re.compile(r'\s+')


def _split_into_sentences(text: str) -> list[str]:
    """Split a paragraph into individual sentences.

    Handles both Devanagari (।) and Latin (. ? !) sentence endings.

    Args:
        text: Input paragraph text.

    Returns:
        List of sentence strings, filtered by minimum length.
    """
    # Normalize Unicode to NFC
    text = unicodedata.normalize("NFC", text)

    # Normalize whitespace
    text = _WHITESPACE_NORM.sub(" ", text).strip()

    if not text:
        return []

    # Split on sentence boundaries
    parts = _SENTENCE_ENDINGS.split(text)

    # Also split on Devanagari danda
    sentences = []
    for part in parts:
        # Split on danda (।) keeping the danda with the sentence
        sub_parts = re.split(r'(?<=।)', part)
        sentences.extend(sub_parts)

    # Clean and filter
    result = []
    for s in sentences:
        s = s.strip()
        # Remove sentences that are too short, pure numbers, or URLs
        if (len(s) >= MIN_SENTENCE_CHARS
                and not re.match(r'^[\d\s\W]+$', s)
                and not s.startswith("http")):
            result.append(s)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    """CLI entry point for the Patrika collector."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Collect sentences from Rajasthan Patrika (patrika.com).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["rajasthan-news", "jaipur-news", "jodhpur-news"],
        help="Sections to scrape",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=50,
        help="Maximum articles per section",
    )
    parser.add_argument(
        "--output",
        default="data/patrika_raw.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY,
        help="Delay between requests in seconds",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    # Check dependencies
    try:
        import bs4
        import requests
    except ImportError:
        print("ERROR: Missing dependencies. Install with:")
        print("  pip install requests beautifulsoup4 lxml")
        sys.exit(1)

    collector = PatrikaCollector(request_delay=args.delay)
    sentences = collector.collect(
        sections=args.sections,
        max_articles_per_section=args.max_articles,
    )

    if not sentences:
        print("No sentences collected. Check your internet connection and try again.")
        sys.exit(1)

    # Save to JSONL
    from pathlib import Path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        for s in sentences:
            record = {
                "text": s.text,
                "source_url": s.source_url,
                "collected_at": s.collected_at.isoformat(),
                "platform": s.platform,
                "sentence_id": s.sentence_id,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✓ Collected {len(sentences)} sentences → {output_path}")
    print(f"\nNext step:")
    print(f"  python run_pipeline.py --manual-data {output_path} --seed 42 --output-dir output/run_001")


if __name__ == "__main__":
    main()
