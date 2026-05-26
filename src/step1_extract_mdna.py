"""
STEP 1 - Extract MD&A sections from Annual Report PDFs
=======================================================
Run this first. It reads every PDF in data/pdfs/ and extracts
the Management Discussion & Analysis section, saving it as a
.txt file in data/raw_mdna/

Requirements:
    pip install pymupdf
"""

import os
import re
import fitz  # PyMuPDF


PDF_FOLDER  = "data/pdfs"
OUT_FOLDER  = "data/raw_mdna"
LOG_FILE    = "outputs/step1_extraction_log.txt"

os.makedirs(OUT_FOLDER, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── Patterns that SIGNAL the start of the MD&A section ─────────────────────
START_PATTERNS = [
    r"management\s+discussion\s+and\s+analysis",
    r"management\s+discussion\s+&\s+analysis",
    r"management['']?s\s+discussion\s+and\s+analysis",
    r"md\s*&\s*a",
    r"mda\b",
    r"management\s+report\s+and\s+analysis",
    r"business\s+overview\s+and\s+outlook",
    r"operating\s+and\s+financial\s+review",
]

# ── Patterns that SIGNAL the END of the MD&A section ────────────────────────
END_PATTERNS = [
    r"corporate\s+governance\s+report",
    r"board\s+of\s+directors['']?\s+report",
    r"directors['']?\s+report",
    r"auditors?['']?\s+report",
    r"independent\s+auditors?['']?\s+report",
    r"standalone\s+financial\s+statements",
    r"consolidated\s+financial\s+statements",
    r"notes\s+to\s+(the\s+)?financial\s+statements",
    r"balance\s+sheet",
    r"profit\s+and\s+loss\s+account",
    r"statement\s+of\s+profit\s+and\s+loss",
    r"cash\s+flow\s+statement",
    r"ten\s+year(s)?\s+financial",
    r"key\s+financial\s+data",
    r"business\s+responsibility\s+report",
    r"secretarial\s+audit\s+report",
]

MIN_MDA_WORDS = 300   # Discard extraction if fewer words found
FALLBACK_PAGES = 20   # Pages to grab if no end-marker found


def compile_patterns(patterns):
    return [re.compile(p, re.IGNORECASE) for p in patterns]


START_RE = compile_patterns(START_PATTERNS)
END_RE   = compile_patterns(END_PATTERNS)


def text_matches(text, compiled_patterns):
    """Return True if any pattern matches in text."""
    for p in compiled_patterns:
        if p.search(text):
            return True
    return False


def is_header_line(line):
    """Heuristic: a heading is short, ALL-CAPS or Title Case, no punctuation."""
    stripped = line.strip()
    if not stripped:
        return False
    words = stripped.split()
    if len(words) > 8:
        return False
    if stripped.isupper():
        return True
    if stripped.istitle():
        return True
    return False


def extract_mdna(pdf_path):
    """
    Open a PDF, find the MD&A section by scanning page text for
    start/end heading keywords, and return the extracted text.
    Returns (text, start_page, end_page) or (None, None, None) on failure.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    # ── Collect page texts (lowercase for matching, original for extraction) ──
    page_texts_lower = []
    page_texts_orig  = []
    for page in doc:
        t = page.get_text("text")
        page_texts_orig.append(t)
        page_texts_lower.append(t.lower())

    # ── Find start page ──────────────────────────────────────────────────────
    start_page = None
    for i, txt in enumerate(page_texts_lower):
        if text_matches(txt, START_RE):
            start_page = i
            break

    if start_page is None:
        # Try looking only at page headers (first 200 chars per page)
        for i, txt in enumerate(page_texts_lower):
            if text_matches(txt[:200], START_RE):
                start_page = i
                break

    if start_page is None:
        return None, None, None  # Could not find MD&A

    # ── Find end page ────────────────────────────────────────────────────────
    end_page = None
    for i in range(start_page + 1, total_pages):
        txt = page_texts_lower[i]
        # Only look at the first 300 chars of a page (likely heading area)
        if text_matches(txt[:300], END_RE):
            end_page = i
            break

    if end_page is None:
        end_page = min(start_page + FALLBACK_PAGES, total_pages)

    # ── Extract text from those pages ────────────────────────────────────────
    extracted_pages = page_texts_orig[start_page:end_page]
    full_text = "\n".join(extracted_pages)

    return full_text, start_page, end_page


def clean_raw_text(text):
    """
    Light cleaning on the raw extracted text:
    - Collapse excessive whitespace / line breaks
    - Remove page-number-only lines
    - Remove lines that are just numbers / symbols
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip lines that are only digits (page numbers)
        if re.fullmatch(r"[\d\s\|\-\.]+", line):
            continue
        # Skip very short noise lines
        if len(line) < 4:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ── Main loop ────────────────────────────────────────────────────────────────
log_lines = []
success   = 0
failed    = 0

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
print(f"Found {len(pdf_files)} PDF files in '{PDF_FOLDER}'\n")

for pdf_file in sorted(pdf_files):
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    company  = pdf_file.replace(".pdf", "")

    print(f"Processing: {pdf_file}")

    text, s, e = extract_mdna(pdf_path)

    if text is None:
        msg = f"FAILED (no MD&A heading found): {pdf_file}"
        print(f"  ✗ {msg}")
        log_lines.append(msg)
        failed += 1
        continue

    text = clean_raw_text(text)
    word_count = len(text.split())

    if word_count < MIN_MDA_WORDS:
        msg = f"FAILED (too short, {word_count} words, pages {s}-{e}): {pdf_file}"
        print(f"  ✗ {msg}")
        log_lines.append(msg)
        failed += 1
        continue

    out_path = os.path.join(OUT_FOLDER, company + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    msg = f"OK pages {s}–{e}, {word_count} words → {out_path}"
    print(f"  ✓ {msg}")
    log_lines.append(f"OK: {pdf_file} | {msg}")
    success += 1

# ── Write log ────────────────────────────────────────────────────────────────
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

print(f"\n{'='*50}")
print(f"Done. Success: {success}  Failed: {failed}")
print(f"Log saved to: {LOG_FILE}")
print(f"Raw MD&A files in: {OUT_FOLDER}")