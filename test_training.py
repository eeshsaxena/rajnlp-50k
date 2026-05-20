#!/usr/bin/env python3
"""Minimal training test to diagnose the crash."""
import sys
import traceback
import logging

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(message)s",
)

print("Step 1: imports")
sys.stdout.flush()

import torch
print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
sys.stdout.flush()

print("Step 2: load transformers")
sys.stdout.flush()
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
print("transformers OK")
sys.stdout.flush()

print("Step 3: load datasets")
sys.stdout.flush()
from datasets import Dataset as HFDataset
import evaluate
print("datasets + evaluate OK")
sys.stdout.flush()

print("Step 4: load corpus")
sys.stdout.flush()
from pathlib import Path
from train_all import load_annotated_corpus
train, val, test = load_annotated_corpus(Path("output/llm_annotations/annotated_corpus.jsonl"))
print(f"Corpus: train={len(train)} val={len(val)} test={len(test)}")
sys.stdout.flush()

print("Step 5: load MuRIL tokenizer")
sys.stdout.flush()
CHECKPOINT = "google/muril-base-cased"
REVISION = "refs/pr/3"  # safetensors version, avoids CVE-2025-32434
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, revision=REVISION)
print("Tokenizer OK")
sys.stdout.flush()

print("Step 6: load MuRIL model")
sys.stdout.flush()
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, revision=REVISION, num_labels=3)
print("Model OK")
sys.stdout.flush()

print("Step 7: build dataset")
sys.stdout.flush()
LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
sample_train = train[:50]
sample_val = val[:10]

def to_hf(sentences):
    return {
        "text": [s.text for s in sentences],
        "label": [LABEL2ID.get(s.sentiment, 1) for s in sentences],
    }

train_hf = HFDataset.from_dict(to_hf(sample_train))
val_hf = HFDataset.from_dict(to_hf(sample_val))

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

train_hf = train_hf.map(tokenize, batched=True)
val_hf = val_hf.map(tokenize, batched=True)
train_hf.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_hf.set_format("torch", columns=["input_ids", "attention_mask", "label"])
print("Dataset built OK")
sys.stdout.flush()

print("Step 8: training args")
sys.stdout.flush()
import numpy as np
metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": metric.compute(predictions=preds, references=labels, average="macro")["f1"]}

args = TrainingArguments(
    output_dir="output/test_run",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=5,
    report_to="none",
    seed=42,
)
print("TrainingArguments OK")
sys.stdout.flush()

print("Step 9: create Trainer")
sys.stdout.flush()
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_hf,
    eval_dataset=val_hf,
    compute_metrics=compute_metrics,
)
print("Trainer OK")
sys.stdout.flush()

print("Step 10: TRAIN")
sys.stdout.flush()
try:
    result = trainer.train()
    print("Training complete:", result)
except Exception as e:
    print("TRAINING FAILED:")
    traceback.print_exc()
sys.stdout.flush()

print("DONE")
