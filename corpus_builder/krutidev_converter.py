"""
Krutidev 010 to Unicode Devanagari converter.

Based on the verified mapping from LTRC, IIIT Hyderabad:
  https://github.com/ltrc/kru2uni
  Copyright 2015, Language Technology Research Center, IIIT Hyderabad
  Authors: Nehal J Wani, Raveesh Motlani
  License: GPL-3.0

Adapted for Python 3 and integrated into the RajNLP-50K pipeline.

Usage:
    from corpus_builder.krutidev_converter import krutidev_to_unicode
    text = krutidev_to_unicode("jktLFkkuh ^Hkk"k")
    # Returns: "राजस्थानी 'भाषा"
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Krutidev 010 → Unicode mapping table (LTRC IIIT Hyderabad, verified)
# ---------------------------------------------------------------------------
# Order matters: longer/more specific patterns must come before shorter ones.

K2U: list[tuple[str, str]] = [
    ('\xf1', '\u0970'),
    ('Q+Z', 'QZ+'),
    ('sas', 'sa'),
    ('aa', 'a'),
    (')Z', '\u0930\u094d\u0926\u094d\u0927'),
    ('ZZ', 'Z'),
    ('\xe5', '\u0966'),
    ('\xd9k', '\u0924\u094d\u0924'),
    ('\xd9', '\u0924\u094d\u0924\u094d'),
    ('\xe4', '\u0915\u094d\u0924'),
    ('=kk', '=k'),
    ('f=k', 'f='),
    ('\xe0', '\u0939\u094d\u0928'),
    ('\xe1', '\u0939\u094d\u092f'),
    ('\xe2', '\u0939\u0943'),
    ('\xe3', '\u0939\u094d\u092e'),
    ('\xbaz', '\u0939\u094d\u0930'),
    ('\xba', '\u0939\u094d'),
    ('\xed', '\u0926\u094d\u0926'),
    ('{k', '\u0915\u094d\u0937'),
    ('{', '\u0915\u094d\u0937\u094d'),
    ('=', '\u0924\u094d\u0930'),
    ('\xab', '\u0924\u094d\u0930\u094d'),
    ('N\xee', '\u091b\u094d\u092f'),
    ('V\xee', '\u091f\u094d\u092f'),
    ('B\xee', '\u0920\u094d\u092f'),
    ('M\xee', '\u0921\u094d\u092f'),
    ('<\xee', '\u0922\u094d\u092f'),
    ('|', '\u0926\u094d\u092f'),
    ('K', '\u091c\u094d\u091e'),
    ('}', '\u0926\u094d\u0935'),
    ('J', '\u0936\u094d\u0930'),
    ('V\xaa', '\u091f\u094d\u0930'),
    ('M\xaa', '\u0921\u094d\u0930'),
    ('<\xaa\xaa', '\u0922\u094d\u0930'),
    ('N\xaa', '\u091b\u094d\u0930'),
    ('\xd8', '\u0915\u094d\u0930'),
    ('\xdd', '\u092b\u094d\u0930'),
    ('nzZ', '\u0930\u094d\u0926\u094d\u0930'),
    ('\xe6', '\u0926\u094d\u0930'),
    ('\xe7', '\u092a\u094d\u0930'),
    ('\xc1', '\u092a\u094d\u0930'),
    ('xz', '\u0917\u094d\u0930'),
    ('#', '\u0930\u0941'),
    (':', '\u0930\u0942'),
    ('v\u201a', '\u0911'),
    ('vks', '\u0913'),
    ('vkS', '\u0914'),
    ('vk', '\u0906'),
    ('v', '\u0905'),
    ('b\xb1', '\u0908\u0902'),
    ('\xc3', '\u0908'),
    ('bZ', '\u0908'),
    ('b', '\u0907'),
    ('m', '\u0909'),
    ('\xc5', '\u090a'),
    (',s', '\u0910'),
    (',', '\u090f'),
    ('_', '\u090b'),
    ('\xf4', '\u0915\u094d\u0915'),
    ('d', '\u0915'),
    ('Dk', '\u0915'),
    ('D', '\u0915\u094d'),
    ('[k', '\u0916'),
    ('[', '\u0916\u094d'),
    ('x', '\u0917'),
    ('Xk', '\u0917'),
    ('X', '\u0917\u094d'),
    ('\xc4', '\u0918'),
    ('?k', '\u0918'),
    ('?', '\u0918\u094d'),
    ('\xb3', '\u0919'),
    ('pkS', '\u091a\u0948'),
    ('p', '\u091a'),
    ('Pk', '\u091a'),
    ('P', '\u091a\u094d'),
    ('N', '\u091b'),
    ('t', '\u091c'),
    ('Tk', '\u091c'),
    ('T', '\u091c\u094d'),
    ('>', '\u091d'),
    ('\xf7', '\u091d\u094d'),
    ('\xa5', '\u091e'),
    ('\xea', '\u091f\u094d\u091f'),
    ('\xeb', '\u091f\u094d\u0920'),
    ('V', '\u091f'),
    ('B', '\u0920'),
    ('\xec', '\u0921\u094d\u0921'),
    ('\xef', '\u0921\u094d\u0922'),
    ('M+', '\u0921\u093c'),
    ('<+', '\u0922\u093c'),
    ('M', '\u0921'),
    ('<', '\u0922'),
    ('.k', '\u0923'),
    ('.', '\u0923\u094d'),
    ('r', '\u0924'),
    ('Rk', '\u0924'),
    ('R', '\u0924\u094d'),
    ('Fk', '\u0925'),
    ('F', '\u0925\u094d'),
    (')', '\u0926\u094d\u0927'),
    ('n', '\u0926'),
    ('/k', '\u0927'),
    ('/', '\u0927\u094d'),
    ('\xcb', '\u0927\u094d'),
    ('\xe8', '\u0927'),
    ('u', '\u0928'),
    ('Uk', '\u0928'),
    ('U', '\u0928\u094d'),
    ('i', '\u092a'),
    ('Ik', '\u092a'),
    ('I', '\u092a\u094d'),
    ('Q', '\u092b'),
    ('\xb6', '\u092b\u094d'),
    ('c', '\u092c'),
    ('Ck', '\u092c'),
    ('C', '\u092c\u094d'),
    ('Hk', '\u092d'),
    ('H', '\u092d\u094d'),
    ('e', '\u092e'),
    ('Ek', '\u092e'),
    ('E', '\u092e\u094d'),
    (';', '\u092f'),
    ('\xb8', '\u092f\u094d'),
    ('j', '\u0930'),
    ('y', '\u0932'),
    ('Yk', '\u0932'),
    ('Y', '\u0932\u094d'),
    ('G', '\u0933'),
    ('o', '\u0935'),
    ('Ok', '\u0935'),
    ('O', '\u0935\u094d'),
    ("'k", '\u0936'),
    ("'", '\u0936\u094d'),
    ('"k', '\u0937'),
    ('"', '\u0937\u094d'),
    ('l', '\u0938'),
    ('Lk', '\u0938'),
    ('L', '\u0938\u094d'),
    ('g', '\u0939'),
    ('\xc8', '\u0940\u0902'),
    ('saz', '\u094d\u0930\u0947\u0902'),
    ('z', '\u094d\u0930'),
    ('\xcc', '\u0926\u094d\u0926'),
    ('\xcd', '\u091f\u094d\u091f'),
    ('\xce', '\u091f\u094d\u0920'),
    ('\xcf', '\u0921\u094d\u0921'),
    ('\xd1', '\u0915\u0943'),
    ('\xd2', '\u092d'),
    ('\xd3', '\u094d\u092f'),
    ('\xd4', '\u0921\u094d\u0922'),
    ('\xd6', '\u091d\u094d'),
    ('\xd8', '\u0915\u094d\u0930'),
    ('\xd9', '\u0924\u094d\u0924\u094d'),
    ('\xdck', '\u0936'),
    ('\xdc', '\u0936\u094d'),
    ('\u201a', '\u0949'),
    ('kas', '\u094b\u0902'),
    ('ks', '\u094b'),
    ('kS', '\u094c'),
    ('\xa1k', '\u093e\u0901'),
    ('ak', 'k\u0902'),
    ('k', '\u093e'),
    ('ah', '\u0940\u0902'),
    ('h', '\u0940'),
    ('aq', '\u0941\u0902'),
    ('q', '\u0941'),
    ('aw', '\u0942\u0902'),
    ('\xa1w', '\u0942\u0901'),
    ('w', '\u0942'),
    ('`', '\u0943'),
    ('as', '\u0947\u0902'),
    ('\xb1s', 's\xb1'),
    ('s', '\u0947'),
    ('aS', '\u0948\u0902'),
    ('S', '\u0948'),
    ('a\xaa', '\u094d\u0930\u0902'),
    ('\xaa', '\u094d\u0930'),
    ('fa', '\u0902f'),
    ('a', '\u0902'),
    ('\xa1', '\u0901'),
    ('%', ':'),
    ('W', '\u0945'),
    ('\u2022', '\u093d'),
    ('\xb7', '\u093d'),
    ('~j', '\u094d\u0930'),
    ('~', '\u094d'),
    ('\\', '?'),
    ('+', '\u093c'),
    ('^', '\u2018'),
    ('*', '\u2019'),
    ('\xde', '\u201c'),
    ('\xdf', '\u201d'),
    ('(', ';'),
    ('\xbc', '('),
    ('\xbd', ')'),
    ('\xbf', '{'),
    ('\xc0', '}'),
    ('\xbe', '='),
    ('A', '\u0964'),
    ('-', '.'),
    ('&', '-'),
    ('\xae', '\u0948\u0902'),
]

_UNICODE_VOWEL_SIGNS = set('\u0905\u0906\u0907\u0908\u0909\u090a\u090f\u0910\u0913\u0914'
                           '\u093e\u093f\u0940\u0941\u0942\u0943\u0947\u0948\u094b\u094c'
                           '\u0902\u0903\u0901\u0945')

_UNICODE_UNATTACHED = set('\u093e\u093f\u0940\u0941\u0942\u0943\u0947\u0948\u094b\u094c'
                          '\u0902\u0903\u0901\u0945')


def krutidev_to_unicode(kru_text: str) -> str:
    """Convert Krutidev 010 text to Unicode Devanagari.

    This is a Python 3 port of the LTRC IIIT Hyderabad kru2uni converter.

    Args:
        kru_text: Text encoded in Krutidev 010 font.

    Returns:
        Unicode Devanagari text, NFC-normalized.
    """
    # Normalize spaces before ्र
    kru_text = kru_text.replace(' \xaa', '\xaa')
    kru_text = kru_text.replace(' ~j', '~j')
    kru_text = kru_text.replace(' z', 'z')

    # Apply all substitutions in order
    for src, dst in K2U:
        kru_text = kru_text.replace(src, dst)

    kru_text = kru_text.replace('\xb1', 'Z\u0902')
    kru_text = kru_text.replace('\xc6', '\u0930\u094df')

    # Fix ि placement: f + consonant → consonant + ि
    misplaced = re.search(r'f(.?)', kru_text)
    while misplaced:
        ch = misplaced.group(1)
        kru_text = kru_text.replace('f' + ch, ch + '\u093f')
        misplaced = re.search(r'f(.?)', kru_text)

    kru_text = kru_text.replace('\xc7', 'fa')
    kru_text = kru_text.replace('\xaf', 'fa')
    kru_text = kru_text.replace('\xc9', '\u0930\u094dfa')

    # Fix fa + consonant → consonant + िं
    misplaced = re.search(r'fa(.?)', kru_text)
    while misplaced:
        ch = misplaced.group(1)
        kru_text = kru_text.replace('fa' + ch, ch + '\u093f\u0902')
        misplaced = re.search(r'fa(.?)', kru_text)

    kru_text = kru_text.replace('\xca', '\u0940Z')

    # Fix ि् + consonant → ् + consonant + ि
    misplaced = re.search('\u093f\u094d(.?)', kru_text)
    while misplaced:
        ch = misplaced.group(1)
        kru_text = kru_text.replace('\u093f\u094d' + ch, '\u094d' + ch + '\u093f')
        misplaced = re.search('\u093f\u094d(.?)', kru_text)

    kru_text = kru_text.replace('\u094dZ', 'Z')

    # Fix र् placement (Z)
    misplaced = re.search(r'(.?)Z', kru_text)
    while misplaced:
        ch = misplaced.group(1)
        idx = kru_text.index(ch + 'Z')
        while idx >= 0 and kru_text[idx] in _UNICODE_VOWEL_SIGNS:
            idx -= 1
            ch = kru_text[idx] + ch
        kru_text = kru_text.replace(ch + 'Z', '\u0930\u094d' + ch)
        misplaced = re.search(r'(.?)Z', kru_text)

    # Clean up illegal matra placements
    for matra in _UNICODE_UNATTACHED:
        kru_text = kru_text.replace(' ' + matra, matra)
        kru_text = kru_text.replace(',' + matra, matra + ',')
        kru_text = kru_text.replace('\u094d' + matra, matra)

    kru_text = kru_text.replace('\u094d\u094d\u0930', '\u094d\u0930')
    kru_text = kru_text.replace('\u094d\u0930\u094d', '\u0930\u094d')
    kru_text = kru_text.replace('\u094d\u094d', '\u094d')
    kru_text = kru_text.replace('\u094d ', ' ')

    return unicodedata.normalize('NFC', kru_text)


def is_likely_krutidev(text: str) -> bool:
    """Detect if text is likely Krutidev-encoded.

    Args:
        text: Text to check.

    Returns:
        True if the text appears to be Krutidev-encoded.
    """
    if not text or len(text) < 10:
        return False

    devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
    total_alpha = devanagari + ascii_alpha

    if total_alpha == 0:
        return False

    return ascii_alpha / total_alpha > 0.65 and devanagari < 10


def convert_pdf_text(text: str) -> str:
    """Convert PDF text to proper Unicode, auto-detecting Krutidev encoding."""
    if is_likely_krutidev(text):
        return krutidev_to_unicode(text)
    return text
