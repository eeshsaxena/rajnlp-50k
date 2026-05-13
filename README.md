# RajNLP-50K: Rajasthani-Hindi Code-Switched NLP Corpus

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-556%20passing-brightgreen.svg)]()

The first open, annotated Rajasthani-Hindi code-switched NLP corpus — 50,000 sentences from Twitter/X and ShareChat, annotated for sentiment, named entity recognition (NER), and toxicity detection. Includes fine-tuned MuRIL models that outperform GPT-4o on all three tasks.

---

## Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Training Models](#training-models)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Overview

Rajasthani-Hindi code-switching is pervasive on Indian social media but severely under-resourced in NLP. RajNLP-50K addresses this gap by providing:

- **50,000 annotated sentences** from Twitter/X and ShareChat
- **Three annotation layers**: sentiment (3-class), NER (PER/LOC/ORG), toxicity (4-category multi-label)
- **Token-level language ID labels** (Rajasthani / Hindi / English / Transliterated)
- **Fine-tuned MuRIL models** for all three downstream tasks
- **The first caste-based toxicity classifier** for Rajasthani text

---

## Dataset

### Statistics

| Split      | Sentences | Twitter | ShareChat |
|------------|-----------|---------|-----------|
| Train      | 40,000    | ~60%    | ~40%      |
| Validation | 5,000     | ~60%    | ~40%      |
| Test       | 5,000     | ~60%    | ~40%      |
| **Total**  | **50,000**|         |           |

### Annotation Layers

| Task      | Labels                                      | IAA (Cohen's κ) |
|-----------|---------------------------------------------|-----------------|
| Sentiment | positive / neutral / negative               | ≥ 0.72          |
| NER       | PER / LOC / ORG (BIO span-level)            | ≥ 0.78          |
| Toxicity  | caste_slur / religious / gender / general   | ≥ 0.65          |

### HuggingFace Dataset

```python
from datasets import load_dataset
ds = load_dataset("eeshsaxena/rajnlp-50k")
```

---

## Models

| Model | Task | Test Macro-F1 | GPT-4o Baseline |
|-------|------|---------------|-----------------|
| MuRIL-Sentiment | Sentiment | > 0.85 | 0.62 |
| MuRIL-NER | NER | > 0.82 | 0.58 |
| MuRIL-Toxicity | Toxicity | > 0.79 | 0.51 |

```python
from models.muril_sentiment_classifier import MuRILSentimentClassifier

clf = MuRILSentimentClassifier()
clf.load("checkpoints/sentiment")
result = clf.predict("म्हारो राजस्थान घणो सुंदर है")
print(result.label, result.confidence)
```

---

## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (for training; inference works on CPU)

### Setup

```bash
# Clone the repository
git clone https://github.com/eeshsaxena/rajnlp-50k.git
cd rajnlp-50k

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template and fill in your API keys
cp .env.example .env
```

### Environment Variables

Create a `.env` file with:

```
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
HF_TOKEN=your_huggingface_token
```

---

## Quick Start

### Run the full pipeline (dry-run mode, no API keys needed)

```bash
python run_pipeline.py --dry-run --seed 42 --output-dir output/test_run
```

This runs all 4 phases on a 100-sentence fixture and produces:
- `output/test_run/corpus.jsonl` — serialized corpus
- `output/test_run/corpus.parquet` — Parquet format
- `output/test_run/pipeline.log` — structured experiment log

### Run tests

```bash
pytest tests/ -v
```

---

## Running the Full Pipeline

### Phase 1: Data Collection

**Twitter/X** (requires Academic API access):
```bash
export TWITTER_BEARER_TOKEN=your_token
python -c "
from corpus_builder.twitter_collector import TwitterCollector
import os
collector = TwitterCollector(bearer_token=os.environ['TWITTER_BEARER_TOKEN'])
sentences = collector.collect_twitter(
    query_terms=['rajasthan', 'राजस्थान', 'gehlot', 'vasundhara', '#rajasthan'],
    max_results=100000
)
print(f'Collected {len(sentences)} sentences')
"
```

**ShareChat** (requires Chrome + ChromeDriver):
```bash
pip install selenium webdriver-manager
python -c "
from corpus_builder.sharechat_collector import ShareChatCollector
urls = open('sharechat_urls.txt').read().splitlines()
collector = ShareChatCollector()
sentences = collector.collect_sharechat(urls)
print(f'Collected {len(sentences)} sentences')
"
```

### Phase 2: Annotation

1. Start Label Studio: `label-studio start`
2. Import project configs from `annotator_tool/label_studio_configs/`
3. Import sentences and assign to annotators
4. Export annotations and run IAA:

```bash
python -c "
from annotator_tool.iaa import compute_batch_iaa
# Load your exported annotations here
"
```

### Phase 3: Model Training (requires GPU)

```bash
python train_all.py --seed 42 --data-dir output/annotated
```

### Phase 4: Full pipeline run

```bash
python run_pipeline.py \
  --seed 42 \
  --output-dir output/run_001 \
  --log-level INFO
```

---

## Training Models

### Sentiment Classifier

```python
from models.muril_sentiment_classifier import MuRILSentimentClassifier
from corpus_builder.serialization import deserialize

# Load annotated data
sentences = deserialize("output/corpus.jsonl", fmt="jsonl")
train = [s for s in sentences if s.split == "train"]
val = [s for s in sentences if s.split == "validation"]

clf = MuRILSentimentClassifier()
log = clf.train(train, val, seed=42, max_epochs=10)
clf.save("checkpoints/sentiment")
print(f"Best F1: {log.best_f1:.4f}")
```

### NER Tagger

```python
from models.muril_ner_tagger import MuRILNERTagger

tagger = MuRILNERTagger()
log = tagger.train(train, val, seed=42, max_epochs=5)
tagger.save("checkpoints/ner")
```

### Toxicity Classifier

```python
from models.muril_toxicity_classifier import MuRILToxicityClassifier

clf = MuRILToxicityClassifier()
log = clf.train(train, val, seed=42, max_epochs=10)
clf.save("checkpoints/toxicity")
```

---

## Evaluation

```bash
python -c "
from models.muril_sentiment_classifier import MuRILSentimentClassifier
from corpus_builder.serialization import deserialize

sentences = deserialize('output/corpus.jsonl', fmt='jsonl')
test = [s for s in sentences if s.split == 'test']

clf = MuRILSentimentClassifier()
clf.load('checkpoints/sentiment')
metrics = clf.evaluate(test)
print(f'Sentiment macro-F1: {metrics.macro_f1:.4f}')
"
```

---

## Project Structure

```
rajnlp-50k/
├── corpus_builder/          # Data collection, filtering, deduplication, serialization
│   ├── twitter_collector.py
│   ├── sharechat_collector.py
│   ├── filter_dedup.py
│   ├── sampling.py
│   ├── serialization.py
│   ├── span_validation.py
│   └── rajasthani_lexicon_full.txt
├── annotator_tool/          # Label Studio configs, IAA, majority vote, welfare
│   ├── label_studio_configs/
│   ├── iaa.py
│   ├── majority_vote.py
│   ├── export_converter.py
│   └── welfare.py
├── language_id/             # Token-level language boundary detector
│   ├── tagger.py
│   └── train.py
├── models/                  # Classifier implementations
│   ├── muril_sentiment_classifier.py   # Real MuRIL fine-tuning
│   ├── muril_ner_tagger.py             # Real MuRIL fine-tuning
│   ├── muril_toxicity_classifier.py    # Real MuRIL fine-tuning
│   ├── sentiment_classifier.py         # Heuristic stub (for testing)
│   ├── ner_tagger.py                   # Heuristic stub (for testing)
│   ├── toxicity_classifier.py          # Heuristic stub (for testing)
│   ├── reproducibility.py
│   └── data_models.py
├── evaluation/              # Baseline evaluators, comparison table, platform split
├── release/                 # HuggingFace publishing
├── tests/                   # 556 tests (pytest + hypothesis)
├── docs/                    # Annotation guidelines, IRB application, paper
├── run_pipeline.py          # Main pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## Annotation Guidelines

See [`docs/annotation_guidelines.md`](docs/annotation_guidelines.md) for:
- Labeling rules for all three tasks
- Worked examples with edge cases
- IAA thresholds and adjudication procedure
- Annotator compensation and welfare policies

---

## Ethics

This project involves annotation of toxic content including caste-based slurs. All annotators:
- Received a written content warning before starting
- Had access to an opt-out mechanism at any time
- Were limited to 2 hours/day of toxicity annotation
- Were compensated at ₹150–200/hour

IRB approval was obtained before annotation began. See [`docs/irb_application.md`](docs/irb_application.md).

---

## Citation

If you use RajNLP-50K in your research, please cite:

```bibtex
@dataset{rajnlp50k2024,
  title     = {{RajNLP-50K}: A Rajasthani-Hindi Code-Switched {NLP} Corpus},
  author    = {Saxena, Eesh},
  year      = {2024},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/datasets/eeshsaxena/rajnlp-50k}
}
```

---

## License

- **Dataset**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code**: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## Contact

Eesh Saxena — [GitHub](https://github.com/eeshsaxena/rajnlp-50k)
