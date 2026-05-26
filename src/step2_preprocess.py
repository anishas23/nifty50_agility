"""
STEP 2 - Preprocess Raw MD&A Text
===================================
Reads every .txt file from data/raw_mdna/
Cleans, tokenises, lemmatises, and saves two outputs:

  data/processed_mdna/<company>.txt   — flat cleaned token string
                                        (used for TF-IDF scoring later)
  data/sentences/<company>.txt        — one sentence per line of tokens
                                        (used to train word2vec)

The paper's exact preprocessing steps (Section 2.2):
  1. Break into sentences
  2. Remove tables, numbers, special characters, stop-words
  3. Remove time words (year, month, quarter) and unit words (crore, percent)
  4. Keep only sentences with ≥ 3 words
  5. Lemmatise
  6. Detect 2- and 3-word phrases (bigrams / trigrams)

Requirements:
    pip install nltk gensim
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt_tab')"
"""

import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models.phrases import Phrases, Phraser

RAW_FOLDER       = "data/raw_mdna"
TOKENS_FOLDER    = "data/processed_mdna"      # flat token strings per company
SENTENCES_FOLDER = "data/sentences"           # sentence-per-line format for word2vec

os.makedirs(TOKENS_FOLDER,    exist_ok=True)
os.makedirs(SENTENCES_FOLDER, exist_ok=True)

# ── Stop-word list ────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))

# Paper explicitly removes these time and unit words
EXTRA_STOPS = {
    "year", "years", "month", "months", "quarter", "quarters",
    "million", "billion", "trillion", "crore", "lakh",
    "percent", "rs", "inr", "usd",
    # Generic report boilerplate
    "company", "companies", "limited", "ltd", "india", "indian",
    "annual", "report", "page", "fy", "fiscal",
    "management", "discussion", "analysis",  # section heading itself
    "also", "including", "however", "therefore", "furthermore",
    "hence", "thus", "please", "refer",
}

STOP_WORDS.update(EXTRA_STOPS)

# ── Lemmatiser ────────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()

# ── Regex helpers ─────────────────────────────────────────────────────────────
RE_NUMBER     = re.compile(r"\d+[\d,\.]*")          # numbers
RE_SPECIAL    = re.compile(r"[^a-z\s]")             # non-alpha chars
RE_WHITESPACE = re.compile(r"\s+")                  # repeated spaces


def normalise_text(text: str) -> str:
    """Lowercase and strip common PDF artefacts before sentence splitting."""
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Remove bullet symbols and common PDF noise
    text = re.sub(r"[•●■▪►‣◦]", " ", text)
    # Collapse dashes used as bullets
    text = re.sub(r"\s[-–—]\s", " ", text)
    return text


def tokenise_sentence(sentence: str):
    """
    Clean one sentence and return a list of lemmatised tokens.
    Returns [] if the cleaned sentence has fewer than 3 tokens.
    """
    sentence = RE_NUMBER.sub(" ", sentence)
    sentence = RE_SPECIAL.sub(" ", sentence)
    sentence = RE_WHITESPACE.sub(" ", sentence).strip()

    tokens = word_tokenize(sentence)
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 2
    ]

    if len(tokens) < 3:
        return []
    return tokens


def process_document(raw_text: str):
    """
    Full pipeline for one document.
    Returns:
        all_token_list  : flat list of all tokens (for TF-IDF)
        sentence_list   : list of token-lists (one per sentence, for word2vec)
    """
    raw_text = normalise_text(raw_text)

    sentences    = sent_tokenize(raw_text)
    sentence_list = []

    for sent in sentences:
        tokens = tokenise_sentence(sent)
        if tokens:
            sentence_list.append(tokens)

    all_token_list = [tok for sent in sentence_list for tok in sent]
    return all_token_list, sentence_list


# ─────────────────────────────────────────────────────────────────────────────
# Pass 1: Process all documents, collect every sentence for bigram training
# ─────────────────────────────────────────────────────────────────────────────
print("Pass 1: Reading and tokenising all documents …")

company_data = {}   # {company_name: (flat_tokens, sentence_list)}
all_sentences = []  # combined sentence list for phrase detection

raw_files = sorted(f for f in os.listdir(RAW_FOLDER) if f.endswith(".txt"))
print(f"  Found {len(raw_files)} raw files.\n")

for fname in raw_files:
    company = fname.replace(".txt", "")
    fpath   = os.path.join(RAW_FOLDER, fname)

    with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
        raw = fh.read()

    flat, sents = process_document(raw)

    if len(flat) < 50:
        print(f"  ⚠ Skipping {company}: too few tokens ({len(flat)})")
        continue

    company_data[company] = (flat, sents)
    all_sentences.extend(sents)
    print(f"  ✓ {company}: {len(flat)} tokens, {len(sents)} sentences")

print(f"\nTotal companies processed : {len(company_data)}")
print(f"Total sentences collected : {len(all_sentences)}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Pass 2: Detect bigrams and trigrams (e.g. "financial_crisis", "interest_rate")
# The paper uses Gensim's Phrases for this (Section 2.2)
# ─────────────────────────────────────────────────────────────────────────────
print("Pass 2: Detecting bigrams …")
bigram_model  = Phrases(all_sentences, min_count=3, threshold=8, delimiter="_")
bigram_phraser = Phraser(bigram_model)

print("Pass 2: Detecting trigrams …")
trigram_model  = Phrases(bigram_phraser[all_sentences], min_count=2, threshold=8, delimiter="_")
trigram_phraser = Phraser(trigram_model)

# ─────────────────────────────────────────────────────────────────────────────
# Pass 3: Apply phrase models, save outputs
# ─────────────────────────────────────────────────────────────────────────────
print("\nPass 3: Applying phrase models and saving …\n")

for company, (flat_tokens, sentence_list) in company_data.items():

    # Apply bigrams then trigrams to each sentence
    phrased_sentences = [
        trigram_phraser[bigram_phraser[sent]]
        for sent in sentence_list
    ]

    # Flat token string for TF-IDF
    flat_phrased = [tok for sent in phrased_sentences for tok in sent]

    # ── Save flat token file ─────────────────────────────────────────────────
    tokens_path = os.path.join(TOKENS_FOLDER, company + ".txt")
    with open(tokens_path, "w", encoding="utf-8") as fh:
        fh.write(" ".join(flat_phrased))

    # ── Save sentence file (one sentence per line) ───────────────────────────
    sents_path = os.path.join(SENTENCES_FOLDER, company + ".txt")
    with open(sents_path, "w", encoding="utf-8") as fh:
        for sent in phrased_sentences:
            fh.write(" ".join(sent) + "\n")

    print(f"  ✓ {company}: {len(flat_phrased)} tokens | {len(phrased_sentences)} sentences")

print("\nDone.")
print(f"Flat token files  → {TOKENS_FOLDER}/")
print(f"Sentence files    → {SENTENCES_FOLDER}/")
print("\nNext step: run step3_lda_topics.py")