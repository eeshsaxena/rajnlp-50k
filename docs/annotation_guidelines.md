# Annotation Guidelines: RajNLP-50K

**Version**: 1.0  
**Project**: RajNLP-50K — Rajasthani-Hindi Code-Switched NLP Corpus  
**Tasks**: Sentiment Analysis · Named Entity Recognition · Toxicity Detection  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [General Instructions](#2-general-instructions)
3. [Task 1: Sentiment Annotation](#3-task-1-sentiment-annotation)
4. [Task 2: Named Entity Recognition](#4-task-2-named-entity-recognition)
5. [Task 3: Toxicity Detection](#5-task-3-toxicity-detection)
6. [Inter-Annotator Agreement](#6-inter-annotator-agreement)
7. [Adjudication Procedure](#7-adjudication-procedure)
8. [Annotator Welfare](#8-annotator-welfare)
9. [Compensation](#9-compensation)
10. [IRB and Ethics](#10-irb-and-ethics)

---

## 1. Introduction

This document defines the labeling rules, worked examples, and edge-case decisions for all three annotation tasks in the RajNLP-50K project. All annotators must read this document in full before beginning annotation.

### What you are annotating

You will label sentences collected from Twitter/X and ShareChat that contain Rajasthani-Hindi code-switching — sentences where speakers mix Rajasthani dialect words with Hindi, English, or transliterated text in a single utterance.

**Example sentence**: "म्हारो घर घणो सुंदर है, BJP ने Jaipur में rally ki"

This sentence mixes:
- Rajasthani: म्हारो (my), घणो (very)
- Hindi: घर (home), सुंदर (beautiful), है (is), में (in)
- English: BJP, rally
- Transliterated: ki (की in Latin script)

---

## 2. General Instructions

- Label each sentence **independently** — do not discuss labels with other annotators before submitting
- Label the sentence **as written** — do not correct spelling or grammar
- If a sentence is completely unintelligible, mark it as **neutral / no entities / non-toxic** and add a note
- If you are unsure, choose the label that best fits the **overall meaning** of the sentence
- You may annotate in any order within a batch

---

## 3. Task 1: Sentiment Annotation

### Labels

| Label | Meaning |
|-------|---------|
| **positive** | The sentence expresses a positive emotion, opinion, or attitude |
| **neutral** | The sentence is factual, informational, or has no clear sentiment |
| **negative** | The sentence expresses a negative emotion, criticism, or complaint |

### Rules

1. Label the **overall sentiment** of the sentence, not individual words
2. Sarcasm counts as **negative** if the intended meaning is negative
3. Questions are usually **neutral** unless they express clear emotion
4. News headlines are usually **neutral**
5. Mixed sentiment → choose the **dominant** sentiment

### Worked Examples

| Sentence | Label | Reason |
|----------|-------|--------|
| म्हारो राजस्थान घणो सुंदर है | positive | Expresses pride and admiration |
| BJP ने Jaipur में rally ki | neutral | Factual statement, no sentiment |
| सरकार कोनी सुणे आम आदमी की बात | negative | Complaint about government |
| Gehlot ji ne achha kaam kiya, par abhi bahut kuch baaki hai | positive | Overall positive despite caveat |
| यो नेता तो बस झूठ बोले है | negative | Criticism |

### Edge Cases

- **Rhetorical questions**: "क्या यही है विकास?" → **negative** (implied criticism)
- **Prayers/blessings**: "भगवान सबका भला करे" → **positive**
- **Announcements**: "कल Jaipur में बारिश होगी" → **neutral**
- **Profanity without clear target**: label based on overall tone

---

## 4. Task 2: Named Entity Recognition

### Entity Types

| Type | Description | Examples |
|------|-------------|---------|
| **PER** | Person names (politicians, celebrities, historical figures) | Gehlot, Vasundhara Raje, Modi, Rahul Gandhi |
| **LOC** | Location names (cities, states, countries, landmarks) | Jaipur, Rajasthan, India, Ajmer |
| **ORG** | Organization names (parties, companies, institutions) | BJP, Congress, INC, Rajasthan University |

### Rules

1. Mark the **full name span** — include titles only if they are part of the name (e.g., "CM Gehlot" → mark "Gehlot" as PER, not "CM")
2. Mark **nested entities** at the most specific level only
3. **Transliterated names** count — "Gehlot" in Latin script is still PER
4. **Abbreviations** count — "BJP" is ORG
5. Zero spans is valid — not every sentence has named entities

### Worked Examples

| Sentence | Entities |
|----------|---------|
| Gehlot ने Jaipur में BJP की rally ki | Gehlot (PER), Jaipur (LOC), BJP (ORG) |
| Modi ji Rajasthan aaye | Modi (PER), Rajasthan (LOC) |
| Congress ne naya neta chuna | Congress (ORG) |
| आज बारिश होगी | (none) |
| Vasundhara Raje ne kaha... | Vasundhara Raje (PER) — full name is one span |

### Edge Cases

- **"CM"**, **"PM"**, **"MLA"** alone → NOT an entity (title without name)
- **"CM Gehlot"** → mark only "Gehlot" as PER
- **"Rajasthan government"** → "Rajasthan" is LOC, "government" is NOT ORG
- **"Rajasthan BJP"** → "Rajasthan BJP" as one ORG span
- **Hashtags**: "#Rajasthan" → LOC (mark without the #)
- **Usernames**: "@GehlotAshok" → PER (mark without the @)

---

## 5. Task 3: Toxicity Detection

> ⚠️ **CONTENT WARNING**: This task involves labeling toxic content including caste-based slurs, religious hatred, gender-based harassment, and general abusive language. Please read Section 8 (Annotator Welfare) before proceeding.

### Labels (multi-label — select all that apply)

| Label | Description |
|-------|-------------|
| **caste_slur** | Contains slurs, derogatory terms, or hate speech targeting a person's caste identity |
| **religious** | Contains religious hatred, incitement, sectarian abuse, or derogatory references to a religion |
| **gender** | Contains gender-based harassment, misogyny, sexist language, or threats targeting gender |
| **general** | Contains general toxic, abusive, threatening, or harassing language not covered above |
| *(none)* | Select no labels if the sentence is non-toxic |

### Rules

1. A sentence can have **zero, one, or multiple** labels
2. Label based on **intent and impact**, not just the presence of a word
3. **Reclaimed language** (a community member using a term about their own community) → use judgment; when in doubt, label it
4. **Indirect toxicity** (implying harm without explicit slurs) → label if the intent is clearly harmful
5. **Quotes of toxic content** (e.g., "he called me [slur]") → label the category of the quoted content

### Worked Examples

| Sentence | Labels | Reason |
|----------|--------|--------|
| यो [caste slur] तो कुछ नहीं कर सकता | caste_slur | Direct caste-based slur |
| काफिर लोगों को यहां नहीं रहना चाहिए | religious | Religious hatred |
| औरत को घर में रहना चाहिए | gender | Sexist/misogynistic statement |
| तू बकवास बंद कर | general | General abusive language |
| Gehlot ne achha kaam kiya | (none) | Non-toxic |
| [slur] और [religious slur] दोनों एक जैसे हैं | caste_slur, religious | Multiple categories |

### Edge Cases

- **Political criticism** without slurs → (none) or general if very aggressive
- **Jokes** that use slurs → label the relevant category
- **Ambiguous Rajasthani words** that could be slurs in context → label if context makes it clear
- **Threats** → general (and religious/gender/caste if applicable)

---

## 6. Inter-Annotator Agreement

After each batch of 500 sentences, IAA is computed using Cohen's Kappa (κ).

| Task | Threshold | Action if below |
|------|-----------|-----------------|
| Sentiment | κ ≥ 0.72 | Batch flagged for adjudication |
| NER | κ ≥ 0.78 | Batch flagged for adjudication |
| Toxicity | κ ≥ 0.65 | Batch flagged for adjudication |

**Final IAA scores** (to be filled after annotation):

| Task | Final κ |
|------|---------|
| Sentiment | TBD |
| NER | TBD |
| Toxicity | TBD |

---

## 7. Adjudication Procedure

When a batch falls below the IAA threshold:

1. The annotation lead reviews all disagreements in the batch
2. A fourth expert annotator labels the disputed sentences independently
3. The expert's label replaces the majority-vote label for disputed sentences
4. The batch is re-evaluated for IAA
5. If IAA is still below threshold, the batch is discussed in a group session

---

## 8. Annotator Welfare

### Content Warning

Before starting the toxicity task, you will see a full content warning describing the nature of the content. You must acknowledge this warning before proceeding.

### Opt-Out

You may withdraw from the toxicity labeling task **at any time, without penalty**. Your sentences will be reassigned to a replacement annotator. To opt out, contact the annotation lead.

### Daily Limit

You will not be assigned more than **2 hours of toxicity sentences per day**. The system enforces this automatically.

### Weekly Check-In

The annotation lead will contact you at least once per week to check on your wellbeing. Please report any distress immediately.

### Support Resources

If you experience distress from the content, please contact:
- **iCall** (India): 9152987821
- **Vandrevala Foundation**: 1860-2662-345 (24/7)

---

## 9. Compensation

| Task | Rate |
|------|------|
| Sentiment annotation | ₹150 per hour |
| NER annotation | ₹150 per hour |
| Toxicity annotation | ₹200 per hour |

Payment is made via bank transfer at the end of each week. You will receive a payment confirmation email.

---

## 10. IRB and Ethics

This project has received ethics approval from [Institution Name] IRB (Protocol #: TBD) before any human annotation of toxic content began.

All annotator data is stored securely and will not be shared with third parties. Annotator identities will not be disclosed in any publication.

---

*Document version 1.0 — Last updated: 2024*
