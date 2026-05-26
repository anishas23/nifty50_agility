"""
STEP 3 - LDA Topic Modelling
=============================
Replicates Section 3.1 of the paper:
  "Exploring the MD&A Section: Which threats are firms concerned about?"

The paper trains LDA on all MD&A sections and identifies 40 topics.
With 30 companies we use 15 topics, which is appropriate for corpus size.

What this script does:
  1. Loads all processed token files
  2. Builds a Gensim dictionary and corpus
  3. Trains LDA (Latent Dirichlet Allocation)
  4. Prints all topics with their top keywords
  5. Shows how much each company's MD&A discusses each topic
  6. Highlights which topic looks like "Threat/Risk" — the key one for agility
  7. Saves topic distributions to outputs/topic_distributions.csv
  8. Saves a topic summary to outputs/lda_topics.txt

Requirements:
    pip install gensim pyLDAvis
"""

import os
import csv
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

TOKENS_FOLDER = "data/processed_mdna"
OUT_TOPICS    = "outputs/lda_topics.txt"
OUT_DIST_CSV  = "outputs/topic_distributions.csv"

os.makedirs("outputs", exist_ok=True)

NUM_TOPICS = 15      # Paper uses 40; 15 is right for 30 companies
NUM_WORDS  = 12      # Top words shown per topic
PASSES     = 30      # Training passes (more = better, slower)


# ── 1. Load documents ─────────────────────────────────────────────────────────
print("Loading processed documents …")

documents    = []
company_names = []

for fname in sorted(os.listdir(TOKENS_FOLDER)):
    if not fname.endswith(".txt"):
        continue
    fpath = os.path.join(TOKENS_FOLDER, fname)
    with open(fpath, "r", encoding="utf-8") as fh:
        tokens = fh.read().strip().split()
    if len(tokens) < 50:
        print(f"  ⚠ Skipping {fname}: too few tokens")
        continue
    documents.append(tokens)
    company_names.append(fname.replace(".txt", ""))
    print(f"  ✓ {fname.replace('.txt','')}: {len(tokens)} tokens")

print(f"\nLoaded {len(documents)} documents.\n")

# ── 2. Build Gensim dictionary ────────────────────────────────────────────────
print("Building vocabulary dictionary …")
dictionary = corpora.Dictionary(documents)

# Remove words that appear in only 1 document (too rare) or
# in more than 70% of documents (too common to be meaningful)
dictionary.filter_extremes(no_below=2, no_above=0.70, keep_n=5000)
print(f"Vocabulary size after filtering: {len(dictionary)} unique tokens\n")

# ── 3. Build BoW corpus ───────────────────────────────────────────────────────
corpus = [dictionary.doc2bow(doc) for doc in documents]

# ── 4. Train LDA ──────────────────────────────────────────────────────────────
print(f"Training LDA with {NUM_TOPICS} topics, {PASSES} passes …")
print("(This may take 1–3 minutes)\n")

lda = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    passes=PASSES,
    alpha="auto",          # Let gensim learn topic concentration
    eta="auto",            # Let gensim learn word distribution
    random_state=42,
    per_word_topics=True,
)

print("LDA training complete.\n")

# ── 5. Compute coherence score (higher = better topic quality) ────────────────
print("Computing coherence score (C_v) …")
coherence_model = CoherenceModel(
    model=lda,
    texts=documents,
    dictionary=dictionary,
    coherence="c_v",
     processes=1,
)
coherence = coherence_model.get_coherence()
print(f"  Coherence Score (C_v): {coherence:.4f}")
print("  (Values above 0.45 indicate reasonable topics)\n")

# ── 6. Print and save all topics ──────────────────────────────────────────────
print("=" * 70)
print("DISCOVERED TOPICS")
print("=" * 70)
print("Look for a topic whose top words relate to: threat, risk, crisis,")
print("challenge, uncertainty, disruption, competition, pressure …")
print("That topic is the 'Threat' topic central to the paper.\n")

topic_lines = [
    f"LDA Topics — {NUM_TOPICS} topics, coherence={coherence:.4f}\n",
    "=" * 70 + "\n"
]

for i in range(NUM_TOPICS):
    topic_words_raw = lda.show_topic(i, topn=NUM_WORDS)
    word_str = "  |  ".join(f"{w} ({p:.3f})" for w, p in topic_words_raw)
    label_hint = ""

    # Auto-flag topics that look threat-related
    threat_indicators = {
        "risk", "challenge", "threat", "crisis", "uncertainty",
        "disruption", "pressure", "competition", "slowdown", "concern",
        "volatile", "headwind", "adverse", "downturn"
    }
    topic_words_set = {w for w, _ in topic_words_raw}
    if topic_words_set & threat_indicators:
        label_hint = "  ◄◄ POSSIBLE THREAT TOPIC"

    line = f"Topic {i:02d}{label_hint}\n  {word_str}"
    print(line)
    topic_lines.append(line + "\n")

with open(OUT_TOPICS, "w", encoding="utf-8") as fh:
    fh.writelines(topic_lines)
print(f"\nTopic list saved → {OUT_TOPICS}\n")

# ── 7. Compute per-company topic distribution ─────────────────────────────────
print("Computing topic weights per company …\n")

# topic_dist[company] = list of (topic_id, proportion) sorted by proportion
rows = []

for company, bow in zip(company_names, corpus):
    topic_probs = dict(lda.get_document_topics(bow, minimum_probability=0.0))
    row = {"company": company}
    for t in range(NUM_TOPICS):
        row[f"topic_{t:02d}"] = round(topic_probs.get(t, 0.0), 4)
    rows.append(row)

# ── Save to CSV ───────────────────────────────────────────────────────────────
fieldnames = ["company"] + [f"topic_{t:02d}" for t in range(NUM_TOPICS)]

with open(OUT_DIST_CSV, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Topic distribution saved → {OUT_DIST_CSV}\n")

# ── 8. Print top topics for each company ──────────────────────────────────────
print("=" * 70)
print("TOP 3 TOPICS PER COMPANY")
print("=" * 70)

for row in rows:
    company = row["company"]
    topic_scores = [(int(k.split("_")[1]), v) for k, v in row.items() if k.startswith("topic_")]
    top3 = sorted(topic_scores, key=lambda x: x[1], reverse=True)[:3]
    top3_str = "  |  ".join(f"Topic {t}: {s:.3f}" for t, s in top3)
    print(f"  {company:<30} {top3_str}")

print("\n" + "=" * 70)
print("ACTION REQUIRED:")
print("  Open outputs/lda_topics.txt")
print("  Find which topic number has threat/risk/crisis/challenge words")
print("  Note that topic number — you need it in step 4 and step 6")
print("=" * 70)
print("\nNext step: run step4_train_word2vec.py")