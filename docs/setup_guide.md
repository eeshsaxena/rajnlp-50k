# Setup Guide: RajNLP-50K

Step-by-step instructions to go from zero to a fully running pipeline.

---

## Step 1: Clone and Install (15 minutes)

```bash
git clone https://github.com/eeshsaxena/rajnlp-50k.git
cd rajnlp-50k

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
```

Verify everything works:
```bash
pytest tests/ -v --tb=short
# Should show: 556 passed
```

---

## Step 2: Apply for Twitter/X Academic API (30 minutes, then wait 1–2 weeks)

1. Go to https://developer.twitter.com/en/portal/petition/academic/is-it-right-for-you
2. Click **Apply for Academic Research access**
3. Fill in:
   - Project name: `RajNLP-50K: Rajasthani-Hindi Code-Switched NLP Corpus`
   - Use case: `Building an annotated NLP corpus for low-resource Rajasthani-Hindi code-switched text for academic research, targeting publication at ACM TALLIP / EMNLP`
   - Institution: your university
4. Wait for approval email
5. Once approved: Developer Portal → Your App → Keys and Tokens → Bearer Token
6. Add to `.env`: `TWITTER_BEARER_TOKEN=your_token`

---

## Step 3: Set Up Label Studio (30 minutes)

```bash
pip install label-studio
label-studio start
# Opens at http://localhost:8080
```

1. Create an account (local, no internet needed)
2. Create 3 projects:
   - **Project 1 — Sentiment**: paste `annotator_tool/label_studio_configs/rajnlp-sentiment.xml` as the labeling config
   - **Project 2 — NER**: paste `annotator_tool/label_studio_configs/rajnlp-ner.xml`
   - **Project 3 — Toxicity**: paste `annotator_tool/label_studio_configs/rajnlp-toxicity.xml`
3. In each project → Settings → Annotation → set **Overlap** = 3, enable **Blind annotation**
4. Create 3 annotator accounts: Settings → Members → Add Member

---

## Step 4: Collect ShareChat Data (2–3 days)

Install Chrome and ChromeDriver:
```bash
pip install selenium webdriver-manager
```

Find 30–50 ShareChat page URLs for Rajasthan content:
- Search Google: `site:sharechat.com rajasthan politics`
- Search Google: `site:sharechat.com राजस्थान`
- Search Google: `site:sharechat.com marwari`

Save URLs to `sharechat_urls.txt` (one per line), then run:
```bash
python -c "
from corpus_builder.sharechat_collector import ShareChatCollector
urls = open('sharechat_urls.txt').read().splitlines()
collector = ShareChatCollector()
sentences = collector.collect_sharechat(urls)
import json, uuid
from datetime import datetime, timezone
with open('data/sharechat_raw.jsonl', 'w') as f:
    for s in sentences:
        f.write(json.dumps({
            'text': s.text,
            'source_url': s.source_url,
            'collected_at': s.collected_at.isoformat(),
            'platform': s.platform,
            'sentence_id': s.sentence_id,
        }, ensure_ascii=False) + '\n')
print(f'Collected {len(sentences)} sentences')
"
```

---

## Step 5: Collect Twitter Data (after API approval)

```bash
python -c "
import os
from corpus_builder.twitter_collector import TwitterCollector

collector = TwitterCollector(bearer_token=os.environ['TWITTER_BEARER_TOKEN'])
sentences = collector.collect_twitter(
    query_terms=[
        'rajasthan', 'राजस्थान', 'gehlot', 'vasundhara',
        '#rajasthan', 'marwari', 'मारवाड़ी', 'jaipur',
        'rajasthani', 'राजस्थानी',
    ],
    max_results=100000,
)
import json
with open('data/twitter_raw.jsonl', 'w') as f:
    for s in sentences:
        f.write(json.dumps({
            'text': s.text,
            'source_url': s.source_url,
            'collected_at': s.collected_at.isoformat(),
            'platform': s.platform,
            'sentence_id': s.sentence_id,
        }, ensure_ascii=False) + '\n')
print(f'Collected {len(sentences)} sentences')
"
```

---

## Step 6: Run the Corpus Pipeline

Once you have raw data:
```bash
python run_pipeline.py \
  --seed 42 \
  --output-dir output/run_001 \
  --log-level INFO
```

This produces:
- `output/run_001/corpus.jsonl` — 50K annotated sentences
- `output/run_001/corpus.parquet` — Parquet format
- `output/run_001/pipeline.log` — experiment log

---

## Step 7: Import to Label Studio and Annotate

```bash
# Export sentences for annotation
python -c "
from corpus_builder.serialization import deserialize
import json

sentences = deserialize('output/run_001/corpus.jsonl', fmt='jsonl')
# Export as Label Studio import format
tasks = [{'data': {'text': s.text, 'sentence_id': s.sentence_id,
                   'source_url': s.source_url, 'platform': s.platform,
                   'collected_at': s.collected_at.isoformat()}}
         for s in sentences]
with open('output/label_studio_import.json', 'w') as f:
    json.dump(tasks, f, ensure_ascii=False, indent=2)
print(f'Exported {len(tasks)} tasks for Label Studio')
"
```

In Label Studio: each project → Import → upload `label_studio_import.json`

---

## Step 8: Train Models (requires GPU)

After annotation is complete and exported:

```bash
# Export from Label Studio → JSON → run converter
python -c "
from annotator_tool.export_converter import convert_sentiment_export
# Load your Label Studio export and convert
"

# Train all models
python train_all.py \
  --seed 42 \
  --data-dir output/run_001 \
  --output-dir checkpoints \
  --log-level INFO
```

Expected training times on A100:
- SentimentClassifier: ~4 hours
- NERTagger: ~6 hours
- ToxicityClassifier: ~8 hours

---

## Step 9: Create HuggingFace Account and Publish

1. Create account at https://huggingface.co
2. Get token at https://huggingface.co/settings/tokens (write access)
3. Add to `.env`: `HF_TOKEN=your_token`
4. Publish:

```bash
python train_all.py \
  --seed 42 \
  --data-dir output/run_001 \
  --output-dir checkpoints \
  --publish \
  --hf-repo-prefix your-username
```

---

## Step 10: Verify Everything

```bash
# Run full test suite
pytest tests/ -v

# Run pipeline smoke test
python run_pipeline.py --dry-run --seed 42 --output-dir output/smoke_test

# Check published dataset
python -c "
from datasets import load_dataset
ds = load_dataset('your-username/rajnlp-50k')
print(ds)
"
```

---

## Troubleshooting

**`CUDA out of memory`**: Reduce batch size in `train_all.py` (try 16 for sentiment, 8 for NER/toxicity)

**`Twitter API 429`**: The collector handles this automatically with exponential backoff. Just wait.

**`Label Studio not starting`**: Try `label-studio start --port 8081` if 8080 is in use

**`seqeval warnings`**: These are benign — seqeval warns about empty sequences which can occur in small test batches
