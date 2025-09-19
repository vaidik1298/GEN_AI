"""
ai_core.py

Legal helper powered by local Ollama LLaMA-3:8B.
Functions:
- simplify_clause(text)
- summarize_document(text)
- classify_clause(text, labels)
- process_dataset(input_path, output_path, glossary_path)
Supports CSV, JSON and PDF input files (PDF text extraction via pdfplumber or PyPDF2).
"""

import requests
import json
import re
import pandas as pd
import csv
from typing import List, Dict, Tuple, Optional
import os

# Optional PDF libraries - used if installed
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

# ---------- CONFIG ----------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3:8b"   # must be pulled with: ollama pull llama3:8b
INPUT_DATASET = "clauses.pdf"   # input dataset (supports .csv, .json, .pdf)
OUTPUT_DATASET = "clauses_out.json"  # output dataset
GLOSSARY_PATH = "CUAD.json"  # optional glossary file (glossary structure expected)
# ----------------------------

# ---------- Ollama wrapper ----------
def _call_ollama(prompt: str, model: str = MODEL, max_tokens: int = 300, temperature: float = 0.0) -> str:
    """
    Calls Ollama model with a prompt and returns the full response text.
    Handles streaming JSON objects line by line.
    """
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        },
        stream=True
    )
    resp.raise_for_status()

    output_chunks = []
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                output_chunks.append(data["response"])
        except json.JSONDecodeError:
            continue

    return "".join(output_chunks).strip()

# ---------- Core functions ----------
def simplify_clause(text: str) -> str:
    prompt = f"Simplify this legal clause into plain English (1–3 sentences):\n\n{text}\n\nSimplified:"
    out = _call_ollama(prompt, max_tokens=180, temperature=0.0)
    return re.split(r"Simplified:|\n\n", out, maxsplit=1)[-1].strip()

def summarize_document(text: str) -> str:
    prompt = f"Summarize this legal text in 3–6 sentences highlighting obligations, risks, and timelines:\n\n{text}\n\nSummary:"
    out = _call_ollama(prompt, max_tokens=300, temperature=0.0)
    return re.split(r"Summary:|\n\n", out, maxsplit=1)[-1].strip()

def classify_clause(text: str, labels: List[str]) -> Tuple[str, Dict]:
    labels_str = ", ".join(labels)
    prompt = (
        f"Classify the following clause into one of these categories: {labels_str}.\n\n"
        f"Clause:\n{text}\n\nAnswer format:\n<Category>\nJustification:"
    )
    out = _call_ollama(prompt, max_tokens=120, temperature=0.0)
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    top_label = lines[0].strip("<>").strip().lower() if lines else ""
    chosen = next((lab for lab in labels if lab.lower() in top_label or top_label in lab.lower()), labels[0]) if labels else ""
    justification = "\n".join(lines[1:]) if len(lines) > 1 else ""
    scores = {lab: (1.0 if lab == chosen else 0.0) for lab in labels} if labels else {}
    return chosen, {"scores": scores, "justification": justification}

# ---------- PDF helpers ----------
def extract_text_from_pdf(path: str) -> str:
    """
    Extract text from a PDF file. Prefer pdfplumber if available, else PyPDF2.
    Returns the concatenated text of all pages.
    """
    text_pages = []
    if pdfplumber:
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text_pages.append(page_text)
            return "\n\n".join(text_pages).strip()
        except Exception:
            # fallback to PyPDF2 if pdfplumber fails
            pass

    if PdfReader:
        try:
            reader = PdfReader(path)
            for p in reader.pages:
                try:
                    page_text = p.extract_text() or ""
                except Exception:
                    page_text = ""
                text_pages.append(page_text)
            return "\n\n".join(text_pages).strip()
        except Exception as e:
            raise RuntimeError(f"PDF extraction failed: {e}")

    raise RuntimeError("No PDF extraction library available. Install 'pdfplumber' or 'PyPDF2'.")

def split_text_into_clauses(text: str) -> List[str]:
    """
    Heuristic splitting of document text into clauses:
    - Prefer paragraphs separated by two or more newlines.
    - If that yields very long or empty list, split on single newlines and group short lines.
    - Trim and discard very short fragments.
    """
    if not text:
        return []

    # Normalize whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    # Try paragraph split first
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]

    # If document seems like one big paragraph, fall back to splitting by lines and grouping
    if len(paragraphs) == 1:
        # split on single newlines
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        # group lines into paragraphs where lines end with sentence terminator or are long
        groups = []
        current = []
        for line in lines:
            current.append(line)
            if len(line) > 120 or re.search(r"[\.!\?]$", line):
                groups.append(" ".join(current))
                current = []
        if current:
            groups.append(" ".join(current))
        paragraphs = [g.strip() for g in groups if g.strip()]

    # final cleanup: remove header line "clause" if present
    if paragraphs and paragraphs[0].strip().lower() == "clause":
        paragraphs = paragraphs[1:]

    # filter out fragments that are too short (e.g., single words)
    clauses = [p for p in paragraphs if len(p.split()) >= 3]

    # if nothing left, return the full text as single clause
    if not clauses and text.strip():
        return [text.strip()]

    return clauses

# ---------- Glossary utilities ----------
def load_glossary(glossary_path: str) -> Dict[str, Dict]:
    """
    Load glossary file (CUAD.json style). Returns a dict mapping normalized term -> entry dict.
    Entry dict includes keys like definition, simplified, category, context.
    """
    if not os.path.exists(glossary_path):
        return {}

    with open(glossary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    terms = {}
    if isinstance(data, dict):
        if "glossary" in data and isinstance(data["glossary"], dict):
            for section in ("legal_terms", "common_phrases"):
                sec = data["glossary"].get(section, {})
                if isinstance(sec, dict):
                    for k, v in sec.items():
                        terms[k.lower()] = v if isinstance(v, dict) else {"original": v}
        else:
            for k, v in data.items():
                if isinstance(v, dict):
                    terms[k.lower()] = v
    return terms

def enrich_with_glossary(clause: str, glossary: Dict[str, Dict]) -> Tuple[str, List[Dict]]:
    """
    Find glossary terms in clause (case-insensitive) and return:
    - annotated_clause: clause with matched terms wrapped as <<TERM>>
    - matches: list of dicts {term, definition, simplified, category, context, start, end}
    """
    if not glossary:
        return clause, []

    candidates = sorted(glossary.keys(), key=lambda s: -len(s))
    lowered = clause.lower()
    matches = []
    spans = []

    for term in candidates:
        term_for_match = term.replace("_", " ")
        pattern = r"\b" + re.escape(term_for_match) + r"\b"
        for m in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            s, e = m.start(), m.end()
            if any(not (e <= a or s >= b) for (a, b) in spans):
                continue
            spans.append((s, e))
            entry = glossary.get(term, {})
            matches.append({
                "term": term_for_match,
                "matched_text": clause[s:e],
                "definition": entry.get("definition") or entry.get("original") or "",
                "simplified": entry.get("simplified", ""),
                "category": entry.get("category", ""),
                "context": entry.get("context", ""),
                "start": s,
                "end": e
            })

    annotated = clause
    # Insert wrappers in reverse order to preserve indices
    for s, e, match in sorted([(m["start"], m["end"], m) for m in matches], key=lambda x: -x[0]):
        pre = annotated[:s]
        mid = annotated[s:e]
        post = annotated[e:]
        annotated = pre + "<<" + mid + ">>" + post

    return annotated, matches

# ---------- Batch processing ----------
def process_dataset(input_path: str, output_path: str, category_labels: Optional[List[str]] = None, glossary_path: Optional[str] = None):
    """
    Process clauses from CSV, JSON or PDF and save structured results.
    If glossary_path points to a valid CUAD.json-like file, glossary matches will be added.
    """
    # load glossary (if provided)
    glossary = {}
    if glossary_path and os.path.exists(glossary_path):
        glossary = load_glossary(glossary_path)
        print(f"Loaded glossary with {len(glossary)} terms from {glossary_path}")

    # read records depending on extension
    records = None
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".csv":
        # defensive CSV reading
        try:
            df = pd.read_csv(input_path)
            if "clause" not in df.columns:
                raise ValueError("CSV must contain a 'clause' column")
            records = df["clause"].astype(str).tolist()
        except Exception:
            try:
                df = pd.read_csv(input_path, engine="python")
                if "clause" in df.columns:
                    records = df["clause"].astype(str).tolist()
                else:
                    df2 = pd.read_csv(input_path, header=None, names=["clause"], sep="\n", engine="python")
                    records = df2["clause"].astype(str).tolist()
            except Exception:
                with open(input_path, "r", encoding="utf-8", errors="replace") as f:
                    raw = f.read().strip()
                lines = raw.splitlines()
                if lines and lines[0].strip().lower() == "clause":
                    lines = lines[1:]
                parts = [p.strip() for p in re.split(r"\n\s*\n", "\n".join(lines)) if p.strip()]
                if not parts:
                    parts = [l.strip() for l in lines if l.strip()]
                records = parts
    elif ext == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            records = [d.get("clause", "") for d in data]
        elif isinstance(data, dict) and "clauses" in data:
            records = [c.get("clause", "") for c in data["clauses"]]
        else:
            # fallback: if top-level is a mapping of terms to definitions, raise helpful error
            raise ValueError("Unsupported JSON format for clauses. Expected list or {'clauses': [...]} structure.")
    elif ext == ".pdf":
        # extract text and split into clauses
        text = extract_text_from_pdf(input_path)
        records = split_text_into_clauses(text)
    else:
        raise ValueError("Unsupported file type. Provide .csv, .json or .pdf")

    if not records:
        raise ValueError("No records found in dataset")

    labels = category_labels or ["payment", "confidentiality", "liability", "termination", "risk", "other"]
    out_list = []

    for i, clause in enumerate(records):
        clause = str(clause).strip()
        annotated, matches = enrich_with_glossary(clause, glossary)
        # LLM calls (guarded)
        try:
            simplified = simplify_clause(clause)
        except Exception as e:
            simplified = ""
            print(f"[LLM] simplify failed for clause {i+1}: {e}")
        try:
            category, meta = classify_clause(clause, labels)
        except Exception as e:
            category, meta = "other", {"scores": {}, "justification": f"LLM classify failed: {e}"}

        out_obj = {
            "original": clause,
            "annotated": annotated,
            "glossary_matches": matches,
            "simplified": simplified,
            "category": category,
            "justification": meta.get("justification", ""),
            "scores": meta.get("scores", {})
        }
        out_list.append(out_obj)
        print(f"[{i+1}/{len(records)}] {category} | matches: {len(matches)} → {simplified[:80]}...")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_list, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved processed dataset to {output_path}")

# ---------- Demo + Auto Dataset Processing ----------
if __name__ == "__main__":
    sample = "The Contractor shall indemnify and hold harmless the Company from any losses or damages arising under this Agreement."

    print("🔹 Demo Run")
    try:
        print("Simplify →", simplify_clause(sample))
        print("\nSummarize →", summarize_document(sample))
        labs = ["payment", "confidentiality", "liability", "termination", "risk"]
        label, meta = classify_clause(sample, labs)
        print("\nClassify →", label, "| Justification:", meta["justification"])
    except Exception as e:
        print("LLM demo failed (is Ollama running?):", e)

    # Process dataset automatically if exists
    if os.path.exists(INPUT_DATASET):
        print(f"\n📂 Processing dataset: {INPUT_DATASET}")
        process_dataset(INPUT_DATASET, OUTPUT_DATASET, glossary_path=GLOSSARY_PATH)
    else:
        print(f"\n⚠️ No dataset found at {INPUT_DATASET}, skipping batch processing.")
