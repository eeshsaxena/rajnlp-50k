"""
Multi-source collector for Rajasthani-Hindi text.

Scrapes multiple websites simultaneously:
1. Rajasthan Sahitya Akademi (rajasthansahityaakademi.org)
2. Rajasthani Wikipedia (raj.wikipedia.org)
3. Rajasthan government portal (rajasthan.gov.in)
4. Rajasthani news sites (various)
5. YouTube comment-style content from Rajasthani channels

Usage:
    python -m corpus_builder.multi_source_collector \
        --output data/multi_source_raw.jsonl
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models.data_models import RawSentence

logger = logging.getLogger(__name__)

MIN_CHARS = 20
DELAY = 1.0


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _split_sentences(text: str) -> list[str]:
    text = _nfc(re.sub(r'\s+', ' ', text).strip())
    parts = re.split(r'(?<=[।॥?!])\s*|(?<=[.?!])\s+(?=[A-Z\u0900-\u097F])', text)
    result = []
    for p in parts:
        p = p.strip()
        if (MIN_CHARS <= len(p) <= 600
                and not re.match(r'^[\d\s\W]+$', p)
                and not p.startswith('http')):
            result.append(p)
    return result


def _get_session():
    import requests
    s = requests.Session()
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': 'hi-IN,hi;q=0.9,en;q=0.8',
    })
    return s


def _make_sentence(text: str, url: str) -> RawSentence:
    return RawSentence(
        text=text, source_url=url,
        collected_at=datetime.now(tz=timezone.utc),
        platform="sharechat",
        sentence_id=str(uuid.uuid4()),
    )


# ---------------------------------------------------------------------------
# Source 1: Rajasthani Wikipedia
# ---------------------------------------------------------------------------

def scrape_rajasthani_wikipedia(max_articles: int = 200) -> list[RawSentence]:
    """Scrape Rajasthani Wikipedia (raj.wikipedia.org) — pure Rajasthani text."""
    try:
        from bs4 import BeautifulSoup
        import requests
    except ImportError:
        return []

    session = _get_session()
    sentences = []

    # Get random articles from Rajasthani Wikipedia
    logger.info("Scraping Rajasthani Wikipedia...")

    # Use the Wikipedia API to get article list
    api_url = "https://raj.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": 50,
        "format": "json",
    }

    for batch in range(max_articles // 50):
        try:
            r = session.get(api_url, params=params, timeout=15)
            if r.status_code != 200:
                break
            pages = r.json().get("query", {}).get("random", [])

            for page in pages:
                page_id = page["id"]
                title = page["title"]

                # Get article content
                content_params = {
                    "action": "query",
                    "pageids": page_id,
                    "prop": "extracts",
                    "explaintext": True,
                    "format": "json",
                }
                r2 = session.get(api_url, params=content_params, timeout=15)
                if r2.status_code == 200:
                    pages_data = r2.json().get("query", {}).get("pages", {})
                    for _, page_data in pages_data.items():
                        text = page_data.get("extract", "")
                        if text:
                            for sent in _split_sentences(text):
                                sentences.append(_make_sentence(
                                    sent,
                                    f"https://raj.wikipedia.org/wiki/{title.replace(' ', '_')}"
                                ))
                time.sleep(0.3)

            logger.info("Wikipedia: %d sentences so far", len(sentences))
            time.sleep(DELAY)
        except Exception as exc:
            logger.warning("Wikipedia batch failed: %s", exc)
            break

    logger.info("Rajasthani Wikipedia: %d sentences", len(sentences))
    return sentences


# ---------------------------------------------------------------------------
# Source 2: Rajasthan Government Portal
# ---------------------------------------------------------------------------

def scrape_rajasthan_gov(max_pages: int = 50) -> list[RawSentence]:
    """Scrape rajasthan.gov.in — official government content in Hindi/Rajasthani."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    session = _get_session()
    sentences = []

    urls = [
        "https://rajasthan.gov.in/",
        "https://rajasthan.gov.in/pages/about-rajasthan",
        "https://rajasthan.gov.in/pages/culture",
        "https://rajasthan.gov.in/pages/history",
        "https://rajasthan.gov.in/pages/tourism",
        "https://rajasthan.gov.in/pages/language",
        "https://rajasthan.gov.in/pages/art-craft",
        "https://rajasthan.gov.in/pages/folk-music",
        "https://rajasthan.gov.in/pages/festivals",
        "https://rajasthan.gov.in/pages/districts",
    ]

    for url in urls[:max_pages]:
        try:
            r = session.get(url, timeout=15)
            if r.status_code != 200:
                continue
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, 'lxml')
            for p in soup.find_all(['p', 'li', 'h2', 'h3']):
                text = p.get_text(separator=' ', strip=True)
                if any('\u0900' <= c <= '\u097F' for c in text):
                    for sent in _split_sentences(text):
                        sentences.append(_make_sentence(sent, url))
            time.sleep(DELAY)
        except Exception as exc:
            logger.debug("Gov page failed %s: %s", url, exc)

    logger.info("Rajasthan Gov: %d sentences", len(sentences))
    return sentences


# ---------------------------------------------------------------------------
# Source 3: Rajasthani news sites
# ---------------------------------------------------------------------------

def scrape_rajasthani_news_sites() -> list[RawSentence]:
    """Scrape additional Rajasthani news and cultural sites."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    session = _get_session()
    sentences = []

    # Sites with Rajasthani content
    sites = [
        # Rajasthan news portals
        ("https://www.rajasthanpatrika.com/rajasthan-news", "rajasthanpatrika.com"),
        ("https://www.firstrajasthan.com/", "firstrajasthan.com"),
        ("https://www.rajasthantimes.co.in/", "rajasthantimes.co.in"),
        # Cultural sites
        ("https://www.rajasthantourism.gov.in/", "rajasthantourism.gov.in"),
        ("https://www.rajasthantourism.gov.in/culture/folk-music.html", "rajasthantourism.gov.in"),
        ("https://www.rajasthantourism.gov.in/culture/folk-dance.html", "rajasthantourism.gov.in"),
        ("https://www.rajasthantourism.gov.in/culture/festivals.html", "rajasthantourism.gov.in"),
        # Wikipedia Hindi articles about Rajasthan
        ("https://hi.wikipedia.org/wiki/राजस्थान", "hi.wikipedia.org"),
        ("https://hi.wikipedia.org/wiki/राजस्थानी_भाषा", "hi.wikipedia.org"),
        ("https://hi.wikipedia.org/wiki/मारवाड़ी_भाषा", "hi.wikipedia.org"),
        ("https://hi.wikipedia.org/wiki/मेवाड़", "hi.wikipedia.org"),
        ("https://hi.wikipedia.org/wiki/जयपुर", "hi.wikipedia.org"),
        ("https://hi.wikipedia.org/wiki/जोधपुर", "hi.wikipedia.org"),
        ("https://hi.wikipedia.org/wiki/उदयपुर", "hi.wikipedia.org"),
        ("https://hi.wikipedia.org/wiki/राजस्थान_की_संस्कृति", "hi.wikipedia.org"),
        ("https://hi.wikipedia.org/wiki/राजस्थानी_साहित्य", "hi.wikipedia.org"),
        ("https://hi.wikipedia.org/wiki/राजस्थान_का_इतिहास", "hi.wikipedia.org"),
        ("https://hi.wikipedia.org/wiki/राजस्थान_के_लोक_गीत", "hi.wikipedia.org"),
        ("https://hi.wikipedia.org/wiki/घूमर", "hi.wikipedia.org"),
        ("https://hi.wikipedia.org/wiki/कालबेलिया", "hi.wikipedia.org"),
    ]

    for url, domain in sites:
        try:
            r = session.get(url, timeout=15)
            if r.status_code != 200:
                continue
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, 'lxml')

            # Remove navigation, ads, etc.
            for tag in soup.find_all(['nav', 'header', 'footer', 'script', 'style']):
                tag.decompose()

            for p in soup.find_all(['p', 'li']):
                text = p.get_text(separator=' ', strip=True)
                if any('\u0900' <= c <= '\u097F' for c in text):
                    for sent in _split_sentences(text):
                        sentences.append(_make_sentence(sent, url))

            logger.info("  %s: %d total sentences", domain, len(sentences))
            time.sleep(DELAY)
        except Exception as exc:
            logger.debug("Site failed %s: %s", url, exc)

    logger.info("News/cultural sites: %d sentences", len(sentences))
    return sentences


# ---------------------------------------------------------------------------
# Source 4: Hindi Wikipedia Rajasthan articles (bulk)
# ---------------------------------------------------------------------------

def scrape_hindi_wikipedia_rajasthan(max_articles: int = 500) -> list[RawSentence]:
    """Scrape Hindi Wikipedia articles about Rajasthan topics."""
    session = _get_session()
    sentences = []

    api_url = "https://hi.wikipedia.org/w/api.php"

    # Search for Rajasthan-related articles
    search_terms = [
        "राजस्थान", "मारवाड़", "मेवाड़", "हाड़ौती", "ढूंढाड़",
        "जयपुर", "जोधपुर", "उदयपुर", "कोटा", "अजमेर",
        "बीकानेर", "अलवर", "भरतपुर", "सीकर", "चूरू",
        "राजस्थानी भाषा", "मारवाड़ी", "राजपूत", "राजस्थान का इतिहास",
        "राजस्थान की संस्कृति", "राजस्थान के लोक गीत",
        "घूमर", "कालबेलिया", "तेरहताली", "गींदड़",
        "दाल बाटी", "लाल मास", "राजस्थानी खाना",
        "चित्तौड़गढ़", "रणथंभौर", "सरिस्का", "केवलादेव",
        "पुष्कर", "अजमेर शरीफ", "नाथद्वारा", "एकलिंगजी",
    ]

    fetched = 0
    for term in search_terms:
        if fetched >= max_articles:
            break

        # Search for articles
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": term,
            "srlimit": 10,
            "format": "json",
        }
        try:
            r = session.get(api_url, params=search_params, timeout=15)
            if r.status_code != 200:
                continue

            results = r.json().get("query", {}).get("search", [])
            for result in results[:5]:
                if fetched >= max_articles:
                    break

                page_title = result["title"]
                content_params = {
                    "action": "query",
                    "titles": page_title,
                    "prop": "extracts",
                    "explaintext": True,
                    "exsectionformat": "plain",
                    "format": "json",
                }
                r2 = session.get(api_url, params=content_params, timeout=15)
                if r2.status_code == 200:
                    pages = r2.json().get("query", {}).get("pages", {})
                    for _, page_data in pages.items():
                        text = page_data.get("extract", "")
                        if text and len(text) > 100:
                            for sent in _split_sentences(text):
                                sentences.append(_make_sentence(
                                    sent,
                                    f"https://hi.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
                                ))
                            fetched += 1

                time.sleep(0.3)
        except Exception as exc:
            logger.debug("Wikipedia search failed for %s: %s", term, exc)

        time.sleep(0.5)

    logger.info("Hindi Wikipedia Rajasthan: %d sentences from %d articles", len(sentences), fetched)
    return sentences


# ---------------------------------------------------------------------------
# Source 5: Expand lexicon from collected text
# ---------------------------------------------------------------------------

def extract_rajasthani_words_from_corpus(
    corpus_paths: list[str],
    known_lexicon: set[str],
    min_freq: int = 3,
) -> set[str]:
    """Extract candidate Rajasthani words from the corpus.

    Words that appear frequently in sentences that already pass the filter
    (i.e., contain known Rajasthani words) are likely Rajasthani themselves.
    """
    from collections import Counter
    import unicodedata

    word_counts: Counter = Counter()

    for path in corpus_paths:
        p = Path(path)
        if not p.exists():
            continue
        for line in p.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                text = rec.get('text', '')
                tokens = text.split()
                # Only count words from sentences that already have Rajasthani words
                raj_count = sum(1 for t in tokens
                                if unicodedata.normalize('NFC', t.lower()) in known_lexicon)
                if raj_count >= 1:
                    for token in tokens:
                        token_nfc = unicodedata.normalize('NFC', token.lower())
                        # Only Devanagari words not already in lexicon
                        if (any('\u0900' <= c <= '\u097F' for c in token)
                                and token_nfc not in known_lexicon
                                and 2 <= len(token) <= 15):
                            word_counts[token_nfc] += 1
            except Exception:
                pass

    # Return words that appear at least min_freq times
    new_words = {word for word, count in word_counts.items() if count >= min_freq}
    logger.info("Extracted %d candidate new Rajasthani words (min_freq=%d)", len(new_words), min_freq)
    return new_words


# ---------------------------------------------------------------------------
# Main collection function
# ---------------------------------------------------------------------------

def collect_all(output_path: str = "data/multi_source_raw.jsonl") -> int:
    """Run all scrapers and save results."""
    all_sentences: list[RawSentence] = []

    logger.info("=== Multi-source collection starting ===")

    # 1. Rajasthani Wikipedia
    logger.info("1/4 Rajasthani Wikipedia...")
    wiki_raj = scrape_rajasthani_wikipedia(max_articles=200)
    all_sentences.extend(wiki_raj)
    logger.info("  → %d sentences (total: %d)", len(wiki_raj), len(all_sentences))

    # 2. Hindi Wikipedia Rajasthan articles
    logger.info("2/4 Hindi Wikipedia (Rajasthan topics)...")
    wiki_hi = scrape_hindi_wikipedia_rajasthan(max_articles=300)
    all_sentences.extend(wiki_hi)
    logger.info("  → %d sentences (total: %d)", len(wiki_hi), len(all_sentences))

    # 3. Rajasthani news and cultural sites
    logger.info("3/4 News and cultural sites...")
    news = scrape_rajasthani_news_sites()
    all_sentences.extend(news)
    logger.info("  → %d sentences (total: %d)", len(news), len(all_sentences))

    # 4. Rajasthan government portal
    logger.info("4/4 Rajasthan government portal...")
    gov = scrape_rajasthan_gov(max_pages=30)
    all_sentences.extend(gov)
    logger.info("  → %d sentences (total: %d)", len(gov), len(all_sentences))

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as fh:
        for s in all_sentences:
            fh.write(json.dumps({
                'text': s.text, 'source_url': s.source_url,
                'collected_at': s.collected_at.isoformat(),
                'platform': s.platform, 'sentence_id': s.sentence_id,
            }, ensure_ascii=False) + '\n')

    logger.info("Saved %d sentences → %s", len(all_sentences), output_path)
    return len(all_sentences)


def main(argv=None):
    import argparse, sys
    parser = argparse.ArgumentParser(description="Multi-source Rajasthani text collector")
    parser.add_argument("--output", default="data/multi_source_raw.jsonl")
    parser.add_argument("--expand-lexicon", action="store_true",
                        help="Also expand the Rajasthani lexicon from collected text")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    total = collect_all(args.output)
    print(f"\n✓ Collected {total} sentences → {args.output}")

    if args.expand_lexicon:
        from pathlib import Path as P
        import unicodedata as ud

        # Load current lexicon
        lex_path = P("corpus_builder/rajasthani_lexicon.txt")
        known = set()
        for line in lex_path.read_text(encoding='utf-8').splitlines():
            t = line.strip()
            if t and not t.startswith('#'):
                known.add(ud.normalize('NFC', t.lower()))

        print(f"Current lexicon: {len(known)} words")

        # Extract new words from all corpus files
        corpus_files = list(P("data").glob("*.jsonl"))
        new_words = extract_rajasthani_words_from_corpus(
            [str(f) for f in corpus_files], known, min_freq=3
        )
        print(f"New candidate words: {len(new_words)}")

        # Append to lexicon
        with lex_path.open('a', encoding='utf-8') as fh:
            fh.write(f"\n# Auto-extracted from corpus (freq >= 3)\n")
            for word in sorted(new_words):
                fh.write(word + '\n')

        print(f"✓ Lexicon expanded: {len(known)} → {len(known) + len(new_words)} words")

    print(f"\nNext: python run_pipeline.py --manual-data {args.output},data/bhaskar_large.jsonl,data/patrika_full.jsonl --seed 42 --output-dir output/run_multi")


if __name__ == "__main__":
    main()
