"""
STEP 5 - Build Threat and Response Dictionaries
=================================================
Replicates Section 3.4 of the paper exactly.

The paper's procedure:
  1. Use 'threat' as seed word → find top 1000 synonyms via cosine similarity
  2. Use 'respond', 'adapt', 'react' combined → find top 1000 synonyms
  3. Manually review and exclude:
        (a) Words that appear in BOTH dictionaries (keep in Response only)
        (b) Words with unclear / unrelated meaning
  4. Final: Threat dict = 574 words, Response dict = 471 words

Since we have a smaller corpus and smaller vocabulary, we:
  - Start with top 200 candidates per dictionary
  - Apply automated filtering (domain-specific exclusion lists)
  - Save candidates for your manual review

THIS SCRIPT:
  A. Generates the raw candidate lists automatically
  B. Applies smart automated filters
  C. Saves final dictionaries to outputs/dictionaries/
  D. Also saves the full candidate list for your manual inspection

MANUAL REVIEW (required after running this script):
  Open outputs/dictionaries/threat_candidates_REVIEW.txt
  Open outputs/dictionaries/response_candidates_REVIEW.txt
  Delete any words that clearly don't belong.
  The script already filters obvious non-threats, but you should review.

Requirements:
    pip install gensim
"""

import os
from gensim.models import Word2Vec

MODEL_PATH  = "models/word2vec_nifty50.model"
DICT_FOLDER = "outputs/dictionaries"

os.makedirs(DICT_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Words to EXCLUDE from Threat dictionary
# These are either too generic, or clearly belong to Response
# ─────────────────────────────────────────────────────────────────────────────
THREAT_BLACKLIST = {
    # Response actions — these belong in Response dict, not Threat
    "respond", "response", "adapt", "adapt", "react", "reaction",
    "address", "mitigate", "manage", "handle", "resolve", "tackle",
    "overcome", "alleviate", "reduce", "lessen", "minimize",
    "hedge", "diversify", "improve", "enhance", "strengthen",
    "develop", "implement", "execute", "deploy", "leverage",
    "transform", "innovate", "invest", "expand", "grow",
    "quickly", "swiftly", "efficiently", "effectively", "promptly",
    "proactively", "responsively", "adequately",
    # Too generic / accounting terms
    "cost", "revenue", "profit", "income", "expense", "tax",
    "asset", "liability", "equity", "dividend", "share", "stock",
    "quarter", "annual", "fiscal", "period", "year", "month",
    # Ambiguous
    "change", "impact", "effect", "result", "outcome", "issue",
    "factor", "condition", "situation", "environment", "nature",
    "level", "degree", "extent", "amount", "number", "rate",
}

# ─────────────────────────────────────────────────────────────────────────────
# Words to EXCLUDE from Response dictionary
# These are threat concepts, or are too generic
# ─────────────────────────────────────────────────────────────────────────────
RESPONSE_BLACKLIST = {
    # Threat words — should be in Threat dict
    "threat", "threaten", "risk", "crisis", "disruption", "challenge",
    "uncertainty", "pressure", "volatility", "competition", "slowdown",
    "decline", "downturn", "recession", "headwind", "adverse", "concern",
    "difficulty", "obstacle", "barrier", "constraint", "problem",
    # Too generic
    "company", "business", "market", "industry", "sector", "segment",
    "product", "service", "customer", "operation", "management",
    "year", "quarter", "period", "growth", "revenue", "profit",
}

# ─────────────────────────────────────────────────────────────────────────────
# Hard-coded seed words for each dictionary
# These are always included if they exist in vocabulary
# ─────────────────────────────────────────────────────────────────────────────
THREAT_SEEDS = {
    "threat", "risk", "crisis", "uncertainty", "disruption",
    "challenge", "competition", "pressure", "volatility", "headwind",
    "slowdown", "downturn", "recession", "pandemic", "geopolitical",
}

RESPONSE_SEEDS = {
    "respond", "adapt", "react", "address", "mitigate",
    "innovate", "transform", "strengthen", "expand", "improve",
    "quickly", "swiftly", "efficiently", "effectively", "proactively",
    "hedge", "diversify", "resilient", "agile", "flexible",
}

# ─────────────────────────────────────────────────────────────────────────────
# Minimum character length for a word to be included
# ─────────────────────────────────────────────────────────────────────────────
MIN_WORD_LEN = 3


def is_valid_token(word: str) -> bool:
    """Return True if word is a valid dictionary candidate."""
    if len(word) < MIN_WORD_LEN:
        return False
    # Skip tokens that are multi-word phrases joined by underscore
    # unless they're meaningful compound terms
    parts = word.split("_")
    if len(parts) > 3:
        return False
    # Skip tokens containing numbers
    if any(ch.isdigit() for ch in word):
        return False
    return True


def get_synonyms(model, seed_words: list, topn: int = 200) -> list:
    """
    Get top synonyms for a list of seed words.
    For multiple seeds, averages their vectors (paper's method).
    Returns list of (word, similarity_score) sorted by score descending.
    """
    # Filter seeds to those in vocabulary
    valid_seeds = [w for w in seed_words if w in model.wv]
    if not valid_seeds:
        print(f"  WARNING: None of the seed words found in vocabulary!")
        return []

    print(f"  Using seeds: {valid_seeds}")

    # Get combined synonyms using positive seeds
    results = model.wv.most_similar(positive=valid_seeds, topn=topn)
    return results


def build_dictionary(candidates: list, blacklist: set, seeds: set,
                     model_vocab: set, label: str) -> list:
    """
    Build final word list from candidates.
    Adds hard seeds, removes blacklisted words.
    Returns sorted list of words.
    """
    final = set()

    # Add hard seeds that are in vocabulary
    for seed in seeds:
        if seed in model_vocab:
            final.add(seed)

    # Add valid candidates
    for word, score in candidates:
        if not is_valid_token(word):
            continue
        if word in blacklist:
            continue
        final.add(word)

    print(f"  {label}: {len(final)} words")
    return sorted(final)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
print("Loading word2vec model …")
model = Word2Vec.load(MODEL_PATH)
vocab = set(model.wv.index_to_key)
print(f"Vocabulary size: {len(vocab)} words\n")

# ── Build Threat dictionary ───────────────────────────────────────────────────
print("=" * 60)
print("BUILDING THREAT DICTIONARY")
print("=" * 60)

threat_seed_list = ["threat", "risk", "crisis", "uncertainty",
                    "disruption", "challenge", "pressure"]
threat_seed_list = [w for w in threat_seed_list if w in vocab]

print(f"Getting top 300 synonyms for threat seeds …")
threat_candidates = get_synonyms(model, threat_seed_list, topn=300)

threat_words = build_dictionary(
    candidates=threat_candidates,
    blacklist=THREAT_BLACKLIST,
    seeds=THREAT_SEEDS,
    model_vocab=vocab,
    label="Threat dictionary"
)

# Remove any overlap: if a word ended up in both, keep in Response only
# (Paper: "more than half excluded due to overlap")
threat_set = set(threat_words)

# ── Build Response dictionary ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BUILDING RESPONSE DICTIONARY")
print("=" * 60)

response_seed_list = ["respond", "adapt", "react",
                      "mitigate", "address", "overcome"]
response_seed_list = [w for w in response_seed_list if w in vocab]

print(f"Getting top 300 synonyms for response seeds …")
response_candidates = get_synonyms(model, response_seed_list, topn=300)

response_words = build_dictionary(
    candidates=response_candidates,
    blacklist=RESPONSE_BLACKLIST,
    seeds=RESPONSE_SEEDS,
    model_vocab=vocab,
    label="Response dictionary"
)

response_set = set(response_words)

# ── Remove overlap (keep in Response, remove from Threat) ────────────────────
overlap = threat_set & response_set
if overlap:
    print(f"\nRemoving {len(overlap)} overlapping words from Threat dict: {sorted(overlap)[:10]} …")
    threat_words = sorted(threat_set - overlap)

# ── Final stats ───────────────────────────────────────────────────────────────
print(f"\nFinal Threat dictionary   : {len(threat_words)} words")
print(f"Final Response dictionary : {len(response_words)} words")
print(f"Overlap removed           : {len(overlap)} words")

# ── Save dictionaries ─────────────────────────────────────────────────────────
threat_path   = os.path.join(DICT_FOLDER, "threat_dictionary.txt")
response_path = os.path.join(DICT_FOLDER, "response_dictionary.txt")

with open(threat_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(threat_words))

with open(response_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(response_words))

print(f"\nSaved → {threat_path}")
print(f"Saved → {response_path}")

# ── Save candidates for manual review ────────────────────────────────────────
review_threat_path   = os.path.join(DICT_FOLDER, "threat_candidates_REVIEW.txt")
review_response_path = os.path.join(DICT_FOLDER, "response_candidates_REVIEW.txt")

with open(review_threat_path, "w", encoding="utf-8") as fh:
    fh.write("# THREAT CANDIDATES — review and delete words that don't fit\n")
    fh.write("# Words already in final dictionary are marked [INCLUDED]\n\n")
    for word, score in threat_candidates[:100]:
        status = "[INCLUDED]" if word in threat_set else "[excluded]"
        fh.write(f"{score:.4f}  {word:<35} {status}\n")

with open(review_response_path, "w", encoding="utf-8") as fh:
    fh.write("# RESPONSE CANDIDATES — review and delete words that don't fit\n")
    fh.write("# Words already in final dictionary are marked [INCLUDED]\n\n")
    for word, score in response_candidates[:100]:
        status = "[INCLUDED]" if word in response_set else "[excluded]"
        fh.write(f"{score:.4f}  {word:<35} {status}\n")

print(f"\nCandidate review files saved:")
print(f"  → {review_threat_path}")
print(f"  → {review_response_path}")

# ── Print top 20 from each (paper's Table 1 equivalent) ──────────────────────
print("\n" + "=" * 60)
print("TOP 20 THREAT WORDS (Table 1 equivalent)")
print("=" * 60)
for i, w in enumerate(threat_words[:20], 1):
    print(f"  {i:>2}. {w}")

print("\n" + "=" * 60)
print("TOP 20 RESPONSE WORDS (Table 1 equivalent)")
print("=" * 60)
for i, w in enumerate(response_words[:20], 1):
    print(f"  {i:>2}. {w}")

print("\n" + "=" * 60)
print("MANUAL REVIEW INSTRUCTIONS")
print("=" * 60)
print("1. Open outputs/dictionaries/threat_candidates_REVIEW.txt")
print("2. Look at [excluded] words — should any of them be INCLUDED?")
print("3. Look at [INCLUDED] words — should any of them be EXCLUDED?")
print("4. If you make changes, manually edit:")
print("   outputs/dictionaries/threat_dictionary.txt")
print("   outputs/dictionaries/response_dictionary.txt")
print("5. Once happy, run step6_compute_scores.py")