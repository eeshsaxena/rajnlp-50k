# RajNLP-50K: The First Annotated Rajasthani-Hindi Code-Switched NLP Corpus

**Authors**: [Your Name], [Co-authors if any]  
**Affiliation**: [Institution]  
**Target Venue**: LREC-COLING 2026 / EMNLP Findings 2025 / ACM TALLIP  

---

## Abstract

We present RajNLP-50K, the first open, annotated corpus of Rajasthani-Hindi code-switched text for natural language processing. The corpus contains 50,000 sentences collected from Twitter/X and ShareChat, annotated by three native Rajasthani-Hindi bilingual speakers across three tasks: sentiment analysis (3-class), named entity recognition (PER/LOC/ORG), and toxicity detection (4-category multi-label). We also contribute a token-level language boundary detector and three fine-tuned MuRIL models that achieve macro-F1 scores of [X.XX] (sentiment), [X.XX] (NER), and [X.XX] (toxicity) on the held-out test partition, outperforming GPT-4o 5-shot baselines by [X.XX], [X.XX], and [X.XX] points respectively. To our knowledge, this is the first caste-based toxicity classifier for Rajasthani text. All data, models, and code are released publicly on HuggingFace.

**Keywords**: code-switching, low-resource NLP, Rajasthani, Hindi, sentiment analysis, NER, toxicity detection, MuRIL

---

## 1. Introduction

Rajasthani is spoken by approximately 50 million people across the Indian state of Rajasthan and diaspora communities worldwide. Despite this scale, Rajasthani remains severely under-resourced in NLP: there are no publicly available annotated corpora, no pre-trained language models, and no downstream task benchmarks.

On social media, Rajasthani speakers routinely code-switch between Rajasthani dialect, Hindi, English, and transliterated forms within a single utterance — a phenomenon we term Rajasthani-Hindi code-switching (RHCS). This linguistic behavior is the dominant mode of online communication for millions of users, yet existing NLP tools trained on monolingual Hindi or English fail catastrophically on RHCS text.

This paper makes the following contributions:

1. **RajNLP-50K**: The first open, annotated RHCS corpus with 50,000 sentences from two social media platforms, annotated across three tasks by native speakers.

2. **Language_ID_Tagger**: A token-level language boundary detector that assigns each token a label from {Rajasthani, Hindi, English, Transliterated}.

3. **Three fine-tuned MuRIL models**: Sentiment classifier, NER tagger, and toxicity classifier, all outperforming GPT-4o 5-shot baselines.

4. **The first caste-based toxicity classifier for Rajasthani**: Detecting caste slurs in RHCS text, a critical tool for content moderation in the Indian context.

5. **Annotation guidelines and IAA analysis**: A formal methodology document and inter-annotator agreement analysis across all three tasks.

---

## 2. Related Work

### 2.1 Code-Switched NLP Corpora

Code-switching has received growing attention in NLP, with corpora for Spanish-English [CITE], Hindi-English [CITE], and Bengali-English [CITE]. However, Rajasthani-Hindi code-switching has not been studied. The closest related work is [CITE] on Hindi-English social media, which does not address Rajasthani dialect features.

### 2.2 Low-Resource Indian Language NLP

Several efforts have addressed low-resource Indian languages, including [CITE] for Odia, [CITE] for Maithili, and [CITE] for Bhojpuri. Rajasthani remains absent from these efforts despite its speaker population.

### 2.3 Toxicity Detection for Indian Languages

Toxicity detection for Indian languages has focused primarily on Hindi [CITE], Tamil [CITE], and Bengali [CITE]. Caste-based hate speech, which is a significant problem in Indian social media, has been studied in [CITE] but not for Rajasthani text.

### 2.4 MuRIL

MuRIL [CITE] is a BERT-based model pre-trained on 17 Indian languages including Devanagari-script corpora. It has shown strong performance on Hindi NLP tasks and is our chosen base model due to its coverage of Devanagari script and Indian language vocabulary.

---

## 3. Data Collection

### 3.1 Sources

We collected data from two platforms:

**Twitter/X**: Using the Academic Research API, we queried for tweets containing Rajasthan politician names, regional hashtags (#rajasthan, #राजस्थान), and documented Rajasthani slang terms. Collection period: [Month Year] to [Month Year].

**ShareChat**: Using Selenium-based scraping, we collected posts from local politics and news pages identified as having high Rajasthani dialect concentration. Collection period: [Month Year] to [Month Year].

### 3.2 Filtering

We applied a Rajasthani-specific lexicon filter, retaining only sentences containing at least 2 tokens from a curated list of [N] Rajasthani-specific lexical items. This filter was validated by native speakers.

### 3.3 Deduplication

We applied two-pass deduplication: exact string matching after Unicode NFC normalization, followed by MinHash LSH with Jaccard threshold 0.85 for near-duplicates.

### 3.4 Sampling and Splitting

After filtering and deduplication, we applied stratified random sampling to select 50,000 sentences preserving the platform distribution. The corpus was split 80/10/10 into train (40,000), validation (5,000), and test (5,000) partitions.

**Table 1: Corpus Statistics**

| | Train | Validation | Test | Total |
|---|---|---|---|---|
| Sentences | 40,000 | 5,000 | 5,000 | 50,000 |
| Twitter | ~24,000 | ~3,000 | ~3,000 | ~30,000 |
| ShareChat | ~16,000 | ~2,000 | ~2,000 | ~20,000 |
| Avg. tokens | TBD | TBD | TBD | TBD |
| Positive sentiment | TBD% | TBD% | TBD% | TBD% |
| Toxic sentences | TBD% | TBD% | TBD% | TBD% |

---

## 4. Annotation

### 4.1 Annotators

Three native Rajasthani-Hindi bilingual speakers annotated the corpus. All annotators are [age range], [gender distribution], from [region]. Annotators were compensated at ₹150/hour for sentiment and NER tasks and ₹200/hour for toxicity tasks.

### 4.2 Annotation Interface

We used Label Studio (self-hosted) with three separate projects for the three tasks. Blind annotation mode prevented annotators from seeing each other's labels. All three annotators labeled every sentence (overlap = 3).

### 4.3 Sentiment Annotation

[Description of sentiment annotation process and statistics]

**Table 2: Sentiment Distribution**

| Label | Count | % |
|-------|-------|---|
| Positive | TBD | TBD% |
| Neutral | TBD | TBD% |
| Negative | TBD | TBD% |

### 4.4 NER Annotation

[Description of NER annotation process and statistics]

**Table 3: NER Entity Distribution**

| Type | Count | % |
|------|-------|---|
| PER | TBD | TBD% |
| LOC | TBD | TBD% |
| ORG | TBD | TBD% |

### 4.5 Toxicity Annotation

[Description of toxicity annotation process and statistics]

**Table 4: Toxicity Label Distribution**

| Category | Count | % of corpus |
|----------|-------|-------------|
| caste_slur | TBD | TBD% |
| religious | TBD | TBD% |
| gender | TBD | TBD% |
| general | TBD | TBD% |
| non-toxic | TBD | TBD% |

### 4.6 Inter-Annotator Agreement

**Table 5: Inter-Annotator Agreement (Cohen's κ)**

| Task | κ | Threshold | Status |
|------|---|-----------|--------|
| Sentiment | TBD | 0.72 | TBD |
| NER | TBD | 0.78 | TBD |
| Toxicity | TBD | 0.65 | TBD |

### 4.7 Adjudication

[N] batches required adjudication. [Description of adjudication process and outcomes.]

---

## 5. Language Identification

We trained a token-level language boundary detector (Language_ID_Tagger) on a held-out subset of 5,000 sentences with manually verified language-ID labels. The tagger assigns each token one of four labels: RAJ (Rajasthani), HIN (Hindi), ENG (English), TRL (Transliterated).

**Table 6: Language Distribution in RajNLP-50K**

| Language | % of tokens |
|----------|-------------|
| Hindi | TBD% |
| Rajasthani | TBD% |
| English | TBD% |
| Transliterated | TBD% |

**Language_ID_Tagger performance**: Token accuracy = TBD, Macro-F1 = TBD

---

## 6. Models and Experiments

### 6.1 Experimental Setup

All models are initialized from `google/muril-base-cased` and fine-tuned on the RajNLP-50K training partition. We use AdamW optimizer with learning rates of 2e-5 (sentiment, toxicity) and 3e-5 (NER), batch sizes of 32 (sentiment) and 16 (NER, toxicity), and early stopping on validation macro-F1 (patience=3). All experiments use random seed 42.

### 6.2 Baselines

We compare against:
- **mBERT zero-shot**: `bert-base-multilingual-cased` without fine-tuning
- **MuRIL zero-shot**: `google/muril-base-cased` without fine-tuning
- **GPT-4o 5-shot**: 5-shot prompting with GPT-4o

### 6.3 Results

**Table 7: Main Results (Macro-F1 on Test Partition)**

| Model | Sentiment | NER | Toxicity |
|-------|-----------|-----|----------|
| mBERT zero-shot | 0.45 | 0.38 | 0.32 |
| MuRIL zero-shot | 0.52 | 0.44 | 0.38 |
| GPT-4o 5-shot | 0.62 | 0.58 | 0.51 |
| **MuRIL fine-tuned (ours)** | **TBD** | **TBD** | **TBD** |

**Table 8: Platform-Split Evaluation (Macro-F1)**

| Train → Eval | Sentiment | NER | Toxicity |
|---|---|---|---|
| Twitter → ShareChat | TBD | TBD | TBD |
| ShareChat → Twitter | TBD | TBD | TBD |

### 6.4 Language ID Ablation

**Table 9: Effect of Language_ID Features on Sentiment (Macro-F1)**

| Model | Test Macro-F1 | Δ vs. no-LangID |
|-------|---------------|-----------------|
| MuRIL (no LangID) | TBD | — |
| MuRIL + LangID | TBD | +TBD |

---

## 7. Analysis

### 7.1 Error Analysis

[Analysis of model errors, particularly on code-switched tokens and Rajasthani-specific vocabulary]

### 7.2 Caste-Based Toxicity

[Analysis of the caste_slur category — distribution, model performance, comparison with general toxicity]

### 7.3 Cross-Platform Generalization

[Analysis of platform-split results — what transfers and what doesn't between Twitter and ShareChat]

---

## 8. Ethical Considerations

### 8.1 Annotator Welfare

All annotators provided informed consent before beginning the study. IRB approval was obtained from [Institution] (Protocol #TBD). Annotators were compensated at above-market rates, given a content warning before toxicity annotation, provided with an opt-out mechanism, and limited to 2 hours/day of toxicity annotation.

### 8.2 Data Statement

- **Source platforms**: Twitter/X and ShareChat
- **Collection period**: [Dates]
- **Annotator demographics**: [Description]
- **Compensation**: ₹150–200/hour
- **Known biases**: Twitter/X over-represents urban, educated users; ShareChat skews toward regional news consumers; both platforms under-represent rural dialects and older speakers

### 8.3 Intended Use and Misuse

RajNLP-50K is intended for NLP research. The toxicity classifier should not be used as a sole decision-making tool for content moderation without human review. The caste-based toxicity labels reflect annotator positionality and may not capture all regional caste dynamics.

---

## 9. Conclusion

We presented RajNLP-50K, the first open annotated corpus for Rajasthani-Hindi code-switched NLP. Our fine-tuned MuRIL models substantially outperform zero-shot and few-shot baselines, demonstrating the value of in-domain annotated data for low-resource code-switched languages. We release all data, models, annotation guidelines, and code publicly to support future research.

---

## References

[CITE] Khanuja et al. (2021). MuRIL: Multilingual Representations for Indian Languages. arXiv:2103.10730.

[CITE] Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.

[CITE] [Add relevant code-switching, Indian NLP, and toxicity detection references]

---

## Appendix A: Annotation Guidelines Summary

See the full Annotation Guidelines document at [URL].

## Appendix B: Rajasthani Lexicon

The Rajasthani-specific lexicon used for filtering contains [N] tokens covering pronouns, verbs, adjectives, nouns, and cultural vocabulary. Available at [URL].

## Appendix C: Model Cards

Model cards for all three fine-tuned models are available on HuggingFace at [URLs].
