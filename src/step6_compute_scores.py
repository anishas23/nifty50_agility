"""
STEP 6 - Compute Threat, Response, and Agility Scores
=======================================================
Replicates Section 3.5 of the paper EXACTLY.

The paper's scoring method (NOT cosine similarity — your original code was wrong):
  - TF.IDF(word_i, document_j) = TF(i,j) * log(N / DF(i))
      TF(i,j)  = raw count of word i in document j
      N        = total number of documents
      DF(i)    = number of documents containing word i at least once
  - Threat Score  = sum of TF-IDF values for all Threat dictionary words in doc
  - Response Score = sum of TF-IDF values for all Response dictionary words in doc
  - Agility Score = Response Score / Threat Score

The paper also computes a 3-year moving average of Agility.
Since we have only 1 year, we report the raw score.

Outputs:
  outputs/agility_scores.csv       — main results table
  outputs/scores_detailed.csv      — threat + response scores too
  outputs/scores_summary.txt       — human-readable summary

Requirements: none beyond standard library + pandas
"""

import os
import math
import csv
import pandas as pd

TOKENS_FOLDER = "data/processed_mdna"
DICT_FOLDER   = "outputs/dictionaries"
OUT_FOLDER    = "outputs"

THREAT_DICT_PATH   = os.path.join(DICT_FOLDER, "threat_dictionary.txt")
RESPONSE_DICT_PATH = os.path.join(DICT_FOLDER, "response_dictionary.txt")
OUT_CSV            = os.path.join(OUT_FOLDER, "agility_scores.csv")
OUT_DETAIL_CSV     = os.path.join(OUT_FOLDER, "scores_detailed.csv")
OUT_SUMMARY        = os.path.join(OUT_FOLDER, "scores_summary.txt")

os.makedirs(OUT_FOLDER, exist_ok=True)


# ── 1. Load dictionaries ──────────────────────────────────────────────────────
def load_dict(path: str) -> set:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dictionary not found: {path}\nRun step5 first.")
    with open(path, "r", encoding="utf-8") as fh:
        words = {line.strip().lower() for line in fh if line.strip() and not line.startswith("#")}
    return words


print("Loading dictionaries …")
threat_dict   = load_dict(THREAT_DICT_PATH)
response_dict = load_dict(RESPONSE_DICT_PATH)
print(f"  Threat   dictionary: {len(threat_dict)} words")
print(f"  Response dictionary: {len(response_dict)} words")

# Verify no overlap (paper requires zero overlap)
overlap = threat_dict & response_dict
if overlap:
    print(f"\n  WARNING: {len(overlap)} words in both dicts: {sorted(overlap)[:5]}")
    print("  Removing from Threat dict (paper's rule: keep in Response)\n")
    threat_dict = threat_dict - overlap

# ── 2. Load all processed documents ──────────────────────────────────────────
print("\nLoading processed documents …")

company_tokens = {}   # {company_name: [token, token, ...]}

for fname in sorted(os.listdir(TOKENS_FOLDER)):
    if not fname.endswith(".txt"):
        continue
    fpath   = os.path.join(TOKENS_FOLDER, fname)
    company = fname.replace(".txt", "")

    with open(fpath, "r", encoding="utf-8") as fh:
        tokens = fh.read().strip().lower().split()

    if len(tokens) < 30:
        print(f"  ⚠ Skipping {company}: too few tokens ({len(tokens)})")
        continue

    company_tokens[company] = tokens
    print(f"  ✓ {company}: {len(tokens)} tokens")

N = len(company_tokens)
print(f"\nTotal documents (N): {N}")

if N == 0:
    raise RuntimeError("No documents loaded. Check data/processed_mdna/ folder.")


# ── 3. Compute Document Frequency (DF) for each word ─────────────────────────
# DF(word) = number of documents in which word appears at least once
print("\nComputing document frequencies …")

# Only need DF for words in our dictionaries (to save time)
all_dict_words = threat_dict | response_dict

df_counts = {word: 0 for word in all_dict_words}

for tokens in company_tokens.values():
    doc_word_set = set(tokens)
    for word in all_dict_words:
        if word in doc_word_set:
            df_counts[word] += 1

# IDF(word) = log(N / DF(word))
# If DF=0 (word never seen), we skip it (contributes 0 to score)
idf = {}
for word in all_dict_words:
    if df_counts[word] > 0:
        idf[word] = math.log(N / df_counts[word])
    else:
        idf[word] = 0.0   # Word not in any doc → contributes nothing


# ── 4. Compute Threat, Response, and Agility scores ──────────────────────────
print("Computing TF-IDF scores …\n")

results = []

for company, tokens in company_tokens.items():

    total_tokens = len(tokens)

    # TF(word, doc) = raw count of word in document
    tf_counts = {}
    for token in tokens:
        if token in all_dict_words:
            tf_counts[token] = tf_counts.get(token, 0) + 1

    # Threat Score = Σ TF(w,d) * IDF(w) for all w in Threat dict
    threat_score = 0.0
    threat_word_contributions = {}
    for word in threat_dict:
        tf  = tf_counts.get(word, 0)
        idf_val = idf.get(word, 0.0)
        contribution = tf * idf_val
        if contribution > 0:
            threat_score += contribution
            threat_word_contributions[word] = round(contribution, 4)

    # Response Score = Σ TF(w,d) * IDF(w) for all w in Response dict
    response_score = 0.0
    response_word_contributions = {}
    for word in response_dict:
        tf  = tf_counts.get(word, 0)
        idf_val = idf.get(word, 0.0)
        contribution = tf * idf_val
        if contribution > 0:
            response_score += contribution
            response_word_contributions[word] = round(contribution, 4)

    # Agility Score = Response Score / Threat Score
    if threat_score > 0:
        agility_score = response_score / threat_score
    else:
        agility_score = 0.0   # If no threats mentioned, agility is undefined

    # Top threat words found in this document
    top_threats   = sorted(threat_word_contributions.items(),
                           key=lambda x: x[1], reverse=True)[:5]
    top_responses = sorted(response_word_contributions.items(),
                           key=lambda x: x[1], reverse=True)[:5]

    results.append({
        "company"             : company,
        "threat_score"        : round(threat_score,   4),
        "response_score"      : round(response_score, 4),
        "agility_score"       : round(agility_score,  4),
        "total_tokens"        : total_tokens,
        "threat_words_found"  : len(threat_word_contributions),
        "response_words_found": len(response_word_contributions),
        "top_threat_words"    : ", ".join(f"{w}({s})" for w, s in top_threats),
        "top_response_words"  : ", ".join(f"{w}({s})" for w, s in top_responses),
    })

    print(f"  {company:<35} | Threat={threat_score:7.3f} | "
          f"Response={response_score:7.3f} | Agility={agility_score:.4f}")


# ── 5. Sort by agility score ──────────────────────────────────────────────────
results.sort(key=lambda x: x["agility_score"], reverse=True)

# Rank companies
for rank, row in enumerate(results, 1):
    row["rank"] = rank


# ── 6. Save main CSV ──────────────────────────────────────────────────────────
main_fields = ["rank", "company", "agility_score", "response_score",
               "threat_score", "total_tokens"]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=main_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)

print(f"\nMain results saved → {OUT_CSV}")

# ── 7. Save detailed CSV ──────────────────────────────────────────────────────
detail_fields = ["rank", "company", "agility_score", "response_score",
                 "threat_score", "total_tokens",
                 "threat_words_found", "response_words_found",
                 "top_threat_words", "top_response_words"]

with open(OUT_DETAIL_CSV, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=detail_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)

print(f"Detailed results saved → {OUT_DETAIL_CSV}")


# ── 8. Print human-readable summary ──────────────────────────────────────────
df = pd.DataFrame(results)

summary_lines = [
    "=" * 70,
    "AGILITY SCORES — NIFTY 50 COMPANIES (1 Year MD&A)",
    "Methodology: Colak & Mai (2026) — TF-IDF weighted dictionary scores",
    "=" * 70,
    "",
    f"Number of companies analysed : {N}",
    f"Threat dictionary size       : {len(threat_dict)} words",
    f"Response dictionary size     : {len(response_dict)} words",
    "",
    "RANKING (most agile → least agile):",
    "-" * 70,
    f"{'Rank':<6} {'Company':<35} {'Agility':>8} {'Response':>10} {'Threat':>8}",
    "-" * 70,
]

for row in results:
    summary_lines.append(
        f"{row['rank']:<6} {row['company']:<35} "
        f"{row['agility_score']:>8.4f} {row['response_score']:>10.3f} "
        f"{row['threat_score']:>8.3f}"
    )

summary_lines += [
    "-" * 70,
    "",
    "DESCRIPTIVE STATISTICS:",
    f"  Mean agility  : {df['agility_score'].mean():.4f}",
    f"  Median agility: {df['agility_score'].median():.4f}",
    f"  Std agility   : {df['agility_score'].std():.4f}",
    f"  Min agility   : {df['agility_score'].min():.4f}",
    f"  Max agility   : {df['agility_score'].max():.4f}",
    "",
    "INTERPRETATION:",
    "  Agility Score = Response Score / Threat Score",
    "  Higher score = firm responds more per unit of threat = MORE AGILE",
    "  A score > 1 means response language exceeds threat language",
    "",
    "NOTES:",
    "  - Scores based on 1 year of MD&A data (paper uses 3-year moving average)",
    "  - Small corpus may affect word2vec dictionary quality",
    "  - Scores are comparable ACROSS companies (cross-sectional)",
    "=" * 70,
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(OUT_SUMMARY, "w", encoding="utf-8") as fh:
    fh.write(summary_text)

print(f"\nSummary saved → {OUT_SUMMARY}")
print("\nNext step: run step7_validation.py")