# Nifty 50 Corporate Agility — Replication Study
## Based on: Colak & Mai (2026), Journal of Corporate Finance

---

## PROJECT STRUCTURE

```
nifty50_agility/
├── data/
│   ├── pdfs/                  ← put your 30 annual report PDFs here
│   ├── raw_mdna/              ← auto-created by step 1
│   ├── processed_mdna/        ← auto-created by step 2
│   ├── sentences/             ← auto-created by step 2
│   └── financial/
│       └── company_financials.csv   ← YOU CREATE THIS (see step 7)
│
├── models/
│   └── word2vec_nifty50.model ← auto-created by step 4
│
├── outputs/
│   ├── dictionaries/
│   │   ├── threat_dictionary.txt
│   │   ├── response_dictionary.txt
│   │   └── *_REVIEW.txt       ← manual review files
│   ├── figures/               ← all plots
│   ├── validation/            ← validation tables
│   ├── agility_scores.csv     ← MAIN RESULT
│   ├── scores_detailed.csv
│   ├── scores_summary.txt
│   ├── lda_topics.txt
│   └── topic_distributions.csv
│
├── step1_extract_mdna.py
├── step2_preprocess.py
├── step3_lda_topics.py
├── step4_train_word2vec.py
├── step5_build_dictionaries.py
├── step6_compute_scores.py
├── step7_validation.py
└── requirements.txt
```

---

## SETUP

### 1. Create virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install pymupdf nltk gensim scikit-learn pandas numpy matplotlib seaborn statsmodels scipy
```

### 3. Download NLTK data (run once)
```bash
python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
"
```

---

## HOW TO RUN — IN ORDER

### STEP 1: Extract MD&A from PDFs
```bash
python step1_extract_mdna.py
```
- Reads every PDF in `data/pdfs/`
- Finds MD&A section automatically using heading keywords
- Saves raw text to `data/raw_mdna/`
- Check `outputs/step1_extraction_log.txt` — any FAILED entries need manual help

**If a company FAILS extraction:**
1. Open the PDF manually
2. Find the MD&A section (usually titled "Management Discussion and Analysis")
3. Copy-paste the text into a .txt file
4. Save as `data/raw_mdna/COMPANYNAME.txt`

---

### STEP 2: Preprocess text
```bash
python step2_preprocess.py
```
- Cleans, lemmatises, finds phrases
- Saves flat token strings to `data/processed_mdna/`
- Saves sentence-per-line format to `data/sentences/`

---

### STEP 3: LDA topic modelling
```bash
python step3_lda_topics.py
```
- Discovers 15 key topics across all MD&A sections
- Prints topics with top keywords
- **Look for the "Threat" topic** — it should have words like: risk, crisis,
  challenge, uncertainty, disruption, pressure, competition
- Note the topic number (you'll refer to it in your report)

---

### STEP 4: Train word2vec
```bash
python step4_train_word2vec.py
```
- Trains word2vec on all sentences from all 30 companies
- Saves model to `models/word2vec_nifty50.model`
- Prints similarity tests — check that 'threat' → crisis, risk, challenge etc.

---

### STEP 5: Build dictionaries
```bash
python step5_build_dictionaries.py
```
- Generates Threat and Response dictionaries
- Saves to `outputs/dictionaries/`
- **MANUAL REVIEW REQUIRED:**
  Open `outputs/dictionaries/threat_candidates_REVIEW.txt`
  Open `outputs/dictionaries/response_candidates_REVIEW.txt`
  For each [INCLUDED] word — does it really belong?
  For each [excluded] word — should it be included?
  Edit the dictionary .txt files directly if needed.

---

### STEP 6: Compute agility scores
```bash
python step6_compute_scores.py
```
- Computes TF-IDF Threat, Response, and Agility scores (paper's exact method)
- Ranks all 30 companies by agility
- Saves results to `outputs/agility_scores.csv`
- Prints summary table

---

### STEP 7: Validation tests
```bash
python step7_validation.py
```
- Runs correlation analysis, regression tests, sector analysis
- Generates figures in `outputs/figures/`

**For full validation, create this file first:**

`data/financial/company_financials.csv`

```
company,board_size,pct_ind_dir,leverage,log_assets,roa,sales_growth,stock_return,sector
RELIANCE,18,55.6,0.42,14.2,0.075,0.12,0.18,Energy
TCS,11,63.6,0.12,13.8,0.23,0.11,0.24,IT
...
```

Data sources:
- **Board data**: BSE annual reports → Corporate Governance Report section
- **Financials**: screener.in (free) → search company → financial ratios
- **Stock returns**: NSE website or `pip install yfinance` then:
  ```python
  import yfinance as yf
  data = yf.download("RELIANCE.NS", start="2023-04-01", end="2024-03-31")
  annual_return = (data['Close'][-1] / data['Close'][0]) - 1
  ```

---

## WHAT EACH SCORE MEANS

| Score | Meaning |
|-------|---------|
| Threat Score | How much the firm discusses threats, risks, challenges in its MD&A |
| Response Score | How much the firm discusses responses, adaptations, mitigations |
| **Agility Score** | **Response / Threat — higher = more agile** |

A score > 1 means the firm's response language exceeds its threat language.
A score < 1 means the firm talks about threats more than it talks about responding.

---

## KEY DIFFERENCES FROM THE PAPER

| Paper | This Replication |
|-------|-----------------|
| 116,353 documents | 30 documents |
| 1997–2020 (24 years) | 1 year |
| US firms (S&P) | Indian firms (Nifty 50) |
| 3-year moving average | Single year score |
| 40 LDA topics | 15 LDA topics |
| Threat dict: 574 words | Smaller (vocabulary limited) |
| FOMC surprise analysis | Can use RBI rate decisions |
| MPU index (US) | India VIX as proxy |

---

## EXPECTED FINDINGS (based on the paper)

If your replication is successful, you should observe:
1. **Sector differences**: IT, FMCG firms likely more agile; PSU/Utilities less agile
2. **Board size**: Negative relationship with agility (larger board = less agile)
3. **Financial constraints**: More agile firms should have lower leverage
4. **Stock returns**: Positive relationship between agility and returns

---

## TROUBLESHOOTING

**"word not in vocabulary" errors in step 5:**
- The word2vec model couldn't learn that word (appeared too rarely)
- This is normal with small corpus — the final dictionaries will be smaller than the paper's

**Step 1 fails to find MD&A in a PDF:**
- Manually open the PDF and check what the MD&A section heading says
- Add that exact heading to `start_keywords` list in step1_extract_mdna.py
- Or manually copy the MD&A text to `data/raw_mdna/COMPANYNAME.txt`

**LDA topics don't look meaningful:**
- Try changing `NUM_TOPICS` in step3 from 15 to 10 or 20
- Run step3 again

**Agility scores all look similar / very low:**
- Check that dictionaries have enough words (open the .txt files)
- Check processed token files have enough content
- Review the candidate lists and add more words manually to dictionaries