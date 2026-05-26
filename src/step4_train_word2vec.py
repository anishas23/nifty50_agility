"""
STEP 4 - Train word2vec Model
==============================
Replicates Section 3.3–3.4 of the paper.

The paper trains word2vec on 116,353 MD&A documents (1997–2020).
We train on 30 Nifty 50 company MD&A files.

IMPORTANT NOTE ON CORPUS SIZE:
  30 documents is a small corpus for word2vec. To partially compensate,
  we use lower min_count and more epochs. The model will still work, but
  synonym quality will be lower than the paper's model. This is expected
  and academically acceptable for a replication study.

Paper hyperparameters used (Section 3.4 footnote):
  - vector_size = 300
  - window      = 5
  - min_count   = 20 (we use 3 due to small corpus)
  - iterations  = 20 (we use 50 for small corpus compensation)

What this saves:
  models/word2vec_nifty50.model  — the trained model
  outputs/word2vec_vocab.txt     — all words the model knows

Requirements:
    pip install gensim
"""

import os
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

SENTENCES_FOLDER = "data/sentences"
MODEL_PATH       = "models/word2vec_nifty50.model"
VOCAB_PATH       = "outputs/word2vec_vocab.txt"

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ── Progress callback ─────────────────────────────────────────────────────────
class EpochLogger(CallbackAny2Vec):
    """Print loss at end of each epoch."""
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        self.epoch += 1
        if self.epoch % 10 == 0 or self.epoch == 1:
            print(f"  Epoch {self.epoch} complete")


# ── Load all sentences from all companies ─────────────────────────────────────
print("Loading sentences from all companies …\n")

all_sentences = []
company_sentence_counts = {}

for fname in sorted(os.listdir(SENTENCES_FOLDER)):
    if not fname.endswith(".txt"):
        continue
    fpath   = os.path.join(SENTENCES_FOLDER, fname)
    company = fname.replace(".txt", "")
    sents   = []

    with open(fpath, "r", encoding="utf-8") as fh:
        for line in fh:
            tokens = line.strip().split()
            if len(tokens) >= 3:
                sents.append(tokens)

    all_sentences.extend(sents)
    company_sentence_counts[company] = len(sents)
    print(f"  {company:<35} {len(sents):>5} sentences")

print(f"\nTotal sentences for training: {len(all_sentences)}\n")

if len(all_sentences) < 100:
    print("ERROR: Too few sentences to train a useful word2vec model.")
    print("Check that step1 and step2 produced non-empty output files.")
    raise SystemExit(1)

# ── Train word2vec ────────────────────────────────────────────────────────────
print("Training word2vec …")
print("  vector_size = 300 (paper: 300)")
print("  window      = 5   (paper: 5)")
print("  min_count   = 3   (paper: 20 — reduced for small corpus)")
print("  epochs      = 50  (paper: 20 — increased for small corpus)")
print("  sg          = 1   (Skip-gram, same as paper)")
print()

logger = EpochLogger()

model = Word2Vec(
    sentences=all_sentences,
    vector_size=300,      # Paper uses 300-dimensional vectors
    window=5,             # Context window size
    min_count=3,          # Min frequency (paper: 20; we lower for small corpus)
    workers=4,            # CPU threads
    sg=1,                 # Skip-gram (better for rare words; paper uses this)
    epochs=50,            # Training epochs
    seed=42,
    compute_loss=True,
    callbacks=[logger],
)

print("\nTraining complete!")

# ── Save the model ────────────────────────────────────────────────────────────
model.save(MODEL_PATH)
print(f"Model saved → {MODEL_PATH}")

# ── Print vocabulary stats ────────────────────────────────────────────────────
vocab_size = len(model.wv)
print(f"\nVocabulary size: {vocab_size} unique tokens")

# Save full vocabulary list
with open(VOCAB_PATH, "w", encoding="utf-8") as fh:
    for word in sorted(model.wv.index_to_key):
        fh.write(word + "\n")
print(f"Vocabulary saved → {VOCAB_PATH}")

# ── Quick similarity tests ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("QUICK SANITY CHECKS")
print("(These show whether the model learned meaningful relationships)")
print("=" * 60)

test_words = ["threat", "risk", "respond", "adapt", "growth",
              "revenue", "market", "digital", "crisis", "strategy"]

for word in test_words:
    if word in model.wv:
        similar = model.wv.most_similar(word, topn=5)
        similar_str = ", ".join(f"{w}({s:.2f})" for w, s in similar)
        print(f"\n  '{word}' → {similar_str}")
    else:
        print(f"\n  '{word}' → NOT IN VOCABULARY (appeared < min_count times)")

# ── Check the most important seeds for dictionaries ──────────────────────────
print("\n" + "=" * 60)
print("KEY SEED WORDS FOR DICTIONARIES (needed in step 5)")
print("=" * 60)

seeds = {
    "Threat dictionary seed"  : "threat",
    "Response dict seed 1"    : "respond",
    "Response dict seed 2"    : "adapt",
    "Response dict seed 3"    : "react",
}

for label, word in seeds.items():
    status = "✓ In vocabulary" if word in model.wv else "✗ NOT in vocabulary"
    print(f"  {label:<30} '{word}' → {status}")

print("\nNext step: run step5_build_dictionaries.py")