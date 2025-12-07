from __future__ import annotations
import os
import re
import pandas as pd
import argparse
import logging
import hashlib
from collections import Counter
from typing import List, Dict, Any, Set, Optional


# PDF extraction backends
try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except Exception:
    _HAS_FITZ = False

try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# -------------------------
# Utilities
# -------------------------
def fingerprint(text: str) -> str:
    h = hashlib.sha1()
    h.update(text.strip().encode("utf-8"))
    return h.hexdigest()

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join([ln for ln in lines if ln]).strip()

def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\.\S+", " ", text)

def remove_file_paths_and_timestamps(text: str) -> str:
    text = re.sub(r"file:///[^\s]+", " ", text)
    text = re.sub(r"[A-Z]:\\[^\s]+", " ", text)
    text = re.sub(r"\[\d{1,2}/\d{1,2}/\d{2,4}.*?\]", " ", text)
    return text

def is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    low = s.lower()
    if re.match(r"^(figure|fig\.|table)\b", low):
        return True
    if re.match(r"^(contents|table of contents)\b", low):
        return True
    if re.match(r"^[\-\*•\s]+$", s):
        return True
    if re.search(r"file://|[A-Z]:\\|\.pdf\b", s):
        return True
    if re.match(r"^\s*\d+\s*$", s):  # lone numbers
        return True
    if len(s) < 3:
        return True
    return False

# -------------------------
# PDF extraction
# -------------------------
def extract_with_fitz(path: str) -> List[str]:
    pages = []
    doc = fitz.open(path)
    for i in range(len(doc)):
        try:
            txt = doc[i].get_text("text") or ""
        except Exception:
            txt = ""
        pages.append(txt)
    doc.close()
    return pages

def extract_with_pdfplumber(path: str) -> List[str]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            try:
                txt = p.extract_text() or ""
            except Exception:
                txt = ""
            pages.append(txt)
    return pages

def extract_pdf_pages(path: str) -> List[str]:
    if _HAS_FITZ:
        pages = extract_with_fitz(path)
        if any(p and len(p.strip()) for p in pages):
            return pages
        logger.warning("PyMuPDF returned empty — falling back to pdfplumber")
    if _HAS_PDFPLUMBER:
        return extract_with_pdfplumber(path)
    raise RuntimeError("Install PyMuPDF or pdfplumber to extract PDF pages.")

# -------------------------
# Header/footer detection 
# -------------------------
def detect_repeated_lines(pages: List[str], min_len: int = 8, threshold_frac: float = 0.25) -> Set[str]:
    firsts = []
    lasts = []
    for p in pages:
        lines = [ln.strip() for ln in p.splitlines() if len(ln.strip()) >= min_len]
        if not lines:
            continue
        firsts.append(lines[0].lower())
        lasts.append(lines[-1].lower())
    combined = firsts + lasts
    counts = Counter(combined)
    n = max(1, len(firsts))
    threshold = max(2, int(n * threshold_frac))
    commons = {ln for ln, c in counts.items() if c >= threshold}
    # remove section-like items (keep "chapter 2" and numbered headings)
    commons = {c for c in commons if not re.match(r'^\d+(\.\d+)*\s+\w+', c)}
    logger.info("Detected %d repeated header/footer candidates", len(commons))
    return commons

def strip_repeated_lines_from_pages(pages: List[str], commons: Set[str]) -> List[str]:
    out = []
    lc = {c.lower() for c in commons}
    for p in pages:
        lines = p.splitlines()
        while lines and lines[0].strip().lower() in lc:
            lines.pop(0)
        while lines and lines[-1].strip().lower() in lc:
            lines.pop(-1)
        out.append("\n".join(lines).strip())
    return out

# -------------------------
# Sentence-aware splitter 
# -------------------------
def sentence_split(text: str) -> List[str]:
    if not text:
        return []
    # protect common abbreviations
    abbrev = r'(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Inc|e\.g|i\.e|etc)\.'
    text = re.sub(abbrev, lambda m: m.group(0).replace('.', '<ABBR>'), text)
    parts = re.split(r'(?<=[\.\?\!])\s+(?=[A-Z0-9"\'])', text)
    parts = [p.replace('<ABBR>', '.') for p in parts]
    return [p.strip() for p in parts if p.strip()]

# -------------------------
# Section-aware chunking
# -------------------------
SECTION_HEADING_RE = re.compile(r'^\s*(chapter\s+\d+|[\d]+(\.[\d]+)*\s+[A-Za-z].*)', flags=re.IGNORECASE)

def split_by_section_markers(page_text: str) -> List[str]:
    lines = page_text.splitlines()
    groups = []
    curr = []
    for ln in lines:
        if SECTION_HEADING_RE.match(ln):
            # start new section if current non-empty
            if curr:
                groups.append("\n".join(curr).strip())
                curr = []
            curr.append(ln)
        else:
            curr.append(ln)
    if curr:
        groups.append("\n".join(curr).strip())
    return [g for g in groups if g.strip()]

def chunk_text_by_sentences(text: str, max_chars: int = 1000, min_chars: int = 120) -> List[str]:
    sents = sentence_split(text)
    chunks = []
    cur = ""
    for s in sents:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip() if cur else s
        else:
            if len(cur) >= min_chars:
                chunks.append(cur.strip())
            cur = s
    if cur and len(cur) >= min_chars:
        chunks.append(cur.strip())
    # merge tiny chunks to neighbor
    merged = []
    for c in chunks:
        if merged and len(c) < int(min_chars * 1.15):
            merged[-1] = (merged[-1] + " " + c).strip()
        else:
            merged.append(c)
    return merged

# -------------------------
# Clean a page 
# -------------------------
def clean_page_text(raw: str) -> str:
    if not raw:
        return ""
    t = raw
    t = remove_urls(t)
    t = remove_file_paths_and_timestamps(t)
    t = normalize_whitespace(t)
    # filter noisy lines but KEEP headings and numbered section lines
    kept = []
    for ln in t.splitlines():
        stripped = ln.strip()
        if not stripped:
            continue
        # keep numbered headings like "2.5 Inheritance"
        if re.match(r'^\d+(\.\d+)*\s+\w+', stripped):
            kept.append(stripped)
            continue
        if is_noise_line(stripped):
            continue
        kept.append(stripped)
    out = "\n".join(kept)
    out = normalize_whitespace(out)
    return out

# -------------------------
# Process a single PDF
# -------------------------
def process_pdf(path: str,
                max_chunk_chars: int = 1000,
                min_chunk_chars: int = 120,
                detect_headers: bool = True) -> List[Dict[str, Any]]:
    logger.info("Processing %s", path)
    pages = extract_pdf_pages(path)
    # skip extremely short pages
    pages = [p for p in pages if p and len(p.strip()) >= 40]
    if not pages:
        logger.warning("No usable pages in %s", path)
        return []

    # detect repeated headers/footers and strip them (but keep headings)
    commons = set()
    if detect_headers:
        commons = detect_repeated_lines(pages)
    if commons:
        pages = strip_repeated_lines_from_pages(pages, commons)

    # truncate at references/bibliography
    for i in range(len(pages) - 1, -1, -1):
        if pages[i] and any(pages[i].strip().lower().startswith(h) for h in ("references", "bibliography", "works cited")):
            logger.info("Truncating at references at page index %d for %s", i, os.path.basename(path))
            pages = pages[:i]
            break

    chunks_out = []
    counter = 0
    for page_idx, raw in enumerate(pages):
        cleaned = clean_page_text(raw)
        if not cleaned or len(cleaned.strip()) < min_chunk_chars:
            continue
        # first try to split by section markers within the page to preserve headings
        sections = split_by_section_markers(cleaned)
        to_chunk = sections if sections else [cleaned]
        for sec in to_chunk:
            pieces = chunk_text_by_sentences(sec, max_chars=max_chunk_chars, min_chars=min_chunk_chars)
            for piece in pieces:
                counter += 1
                chunks_out.append({
                    "chunk_id": f"{os.path.basename(path)}_c{counter:06d}",
                    "source_file": os.path.basename(path),
                    "page_number": page_idx + 1,
                    "text": piece,
                    "chunk_length": len(piece),
                    "type": "educational_content"
                })

    # dedupe near-exact duplicates (fingerprint)
    seen = set()
    unique = []
    for r in chunks_out:
        fp = fingerprint(r["text"])
        if fp in seen:
            continue
        seen.add(fp)
        unique.append(r)

    logger.info("Created %d unique chunks for %s", len(unique), os.path.basename(path))
    return unique

# -------------------------
# Build KB for a directory
# -------------------------
def build_knowledge_base(textbooks_dir: str, output_csv: str,
                             max_chunk_chars: int = 1000, min_chunk_chars: int = 120,
                             detect_headers: bool = True) -> None:
    if not os.path.isdir(textbooks_dir):
        raise RuntimeError(f"textbooks dir not found: {textbooks_dir}")
    pdfs = [f for f in os.listdir(textbooks_dir) if f.lower().endswith(".pdf")]
    if not pdfs:
        raise RuntimeError("No pdf files found in textbooks dir")
    all_chunks: List[Dict[str, Any]] = []
    for p in sorted(pdfs):
        path = os.path.join(textbooks_dir, p)
        try:
            ch = process_pdf(path, max_chunk_chars, min_chunk_chars, detect_headers)
            all_chunks.extend(ch)
        except Exception as e:
            logger.exception("Failed to process %s: %s", path, e)

    if not all_chunks:
        logger.error("No chunks created.")
        return

    df = pd.DataFrame(all_chunks)
    # filter by sane length
    df = df[df['chunk_length'] >= min_chunk_chars]
    df = df[df['chunk_length'] <= max_chunk_chars * 4]  # safety upper bound

    # re-number chunk ids sequentially per file
    def renumber(df_in: pd.DataFrame) -> pd.DataFrame:
        out_rows = []
        for fname, group in df_in.groupby("source_file"):
            for i, (_, row) in enumerate(group.iterrows(), start=1):
                r = row.copy()
                r['chunk_id'] = f"{fname}_c{i:06d}"
                out_rows.append(r)
        return pd.DataFrame(out_rows)
    df = renumber(df)

    # final CSV
    df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info("Saved cleaned KB CSV: %s (%d rows)", output_csv, len(df))

# -------------------------
# CLI Helper Wrappers
# -------------------------
def clean_text(text: str) -> str:
    """ Compatibility wrapper used by some helper scripts that call `clean_text`."""
    return clean_page_text(text)


def split_to_sentences(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """ Compatibility wrapper for older helper that expects `split_to_sentences`."""
    min_chars = max(120, int(chunk_size * 0.12))
    return chunk_text_by_sentences(text, max_chars=chunk_size, min_chars=min_chars)

# ----------------------------------------------------
# CORE PROCESSING FUNCTION FOR STREAMLIT UPLOADER
# ----------------------------------------------------
def process_documents(file_paths: list, output_csv: str = "educational_knowledge_base.csv", 
                      max_chunk_chars: int = 1000, min_chunk_chars: int = 120):
    """
    Processes a list of PDF files, chunks the content, and updates/appends 
    to the knowledge base CSV, skipping files already present.
    """
    all_chunks = []
    
    # 1. Load existing data to preserve old chunks and track existing sources
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        existing_sources = set(df_existing['source_file'].unique())
        # Append existing chunks to the new list
        all_chunks.extend(df_existing.to_dict('records'))
    else:
        existing_sources = set()

    
    # 2. Process NEW files only
    newly_processed_files = []
    
    # Use existing chunk IDs from the DataFrame for robust renumbering
    existing_df = pd.DataFrame(all_chunks)
    
    # Temporary holder for new chunks created during this run
    new_chunks_temp = [] 

    for file_path in file_paths:
        source_file = os.path.basename(file_path)
        
        # Check if file name already exists (case sensitive)
        if source_file in existing_sources:
            logger.info(f"Skipping {source_file}: already processed.")
            continue
        
        try:
            # Use the existing robust process_pdf function
            new_file_chunks = process_pdf(
                path=file_path,
                max_chunk_chars=max_chunk_chars, 
                min_chunk_chars=min_chunk_chars,
                detect_headers=True
            )
            
            new_chunks_temp.extend(new_file_chunks)
            newly_processed_files.append(source_file)
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            continue

    if not newly_processed_files:
        return len(existing_df) # Return count of existing chunks if no new files were processed

    # 3. Combine existing data with newly processed data
    df_combined = pd.concat([existing_df, pd.DataFrame(new_chunks_temp)], ignore_index=True)
    
    # 4. Filter and re-number chunk ids sequentially per file (MANDATORY for KB integrity)
    df_combined = df_combined[df_combined['chunk_length'] >= min_chunk_chars]
    df_combined = df_combined[df_combined['chunk_length'] <= max_chunk_chars * 4] 

    def renumber(df_in: pd.DataFrame) -> pd.DataFrame:
        out_rows = []
        for fname, group in df_in.groupby("source_file"):
            # Ensure index starts from 1 for the renumbering logic
            for i, (_, row) in enumerate(group.iterrows(), start=1):
                r = row.copy()
                r['chunk_id'] = f"{fname}_c{i:06d}"
                out_rows.append(r)
        return pd.DataFrame(out_rows)

    df_final = renumber(df_combined)

    # 5. Final save/overwrite the knowledge base CSV
    df_final.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info("Successfully updated KB CSV: %s (%d total rows)", output_csv, len(df_final))
    
    return len(df_final)

# -------------------------
# CLI Main execution
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--textbooks-dir", default="textbooks")
    p.add_argument("--output", default="educational_knowledge_base.csv")
    p.add_argument("--max-chars", type=int, default=1000)
    p.add_argument("--min-chars", type=int, default=120)
    p.add_argument("--no-detect-headers", dest="detect_headers", action="store_false")
    return p.parse_args()

def main():
    args = parse_args()
    build_knowledge_base(args.textbooks_dir, args.output, max_chunk_chars=args.max_chars,
                             min_chunk_chars=args.min_chars, detect_headers=args.detect_headers)
    
# Commented out the main execution block to make the file importable
# if __name__ == "__main__":
#     main()
