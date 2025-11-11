from flask import Flask, request, jsonify
import pandas as pd
import os, re
from typing import List

app = Flask(__name__)

# ---------- Config ----------
CSV_PATH = os.getenv("ITEMS_CSV_PATH", "data/Items.csv")
SEARCHABLE_FIELDS_DEFAULT = [
    "Stock Code",
    "Name",
    "Short description",
    "Marketing description",
    "Applications",
    "Benefits",
]
STOP_WORDS = {"the","and","for","of","a","in","to","on","at","with","by"}

# Try common datasheet column names (first one found will be used when highlighting)
DATASHEET_CANDIDATES = [
    "Data sheet","Data Sheet","Datasheet","Datasheet URL","Data sheet URL","DataSheet","Data_Sheet"
]

# ---------- Helpers ----------
def normalize_code(s: str) -> str:
    """Uppercase and remove spaces, dashes, dots, underscores and slashes."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"[ \-._/]+", "", s).upper()

def tokenize(q: str) -> List[str]:
    q = q.lower()
    q = re.sub(r"[^a-z0-9\s,]+", " ", q)
    parts = [p for p in re.split(r"[,\s]+", q) if p]
    # dedupe and stop-word filter (preserve order)
    seen, out = set(), []
    for t in parts:
        if t in STOP_WORDS: 
            continue
        if t not in seen:
            out.append(t); seen.add(t)
    return out

def clamp(n, lo, hi):
    return max(lo, min(hi, n))

# ---------- Load CSV + build index ----------
try:
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
except Exception as e:
    raise RuntimeError(f"Failed to load items file from {CSV_PATH}: {e}")

# Ensure expected columns exist
cols_lower = {c.lower(): c for c in df.columns}
if "stock code".lower() not in cols_lower:
    raise RuntimeError(f"'Stock Code' column not found. Columns present: {list(df.columns)}")

# Decide searchable fields (only keep those that actually exist)
SEARCHABLE_FIELDS = [c for c in SEARCHABLE_FIELDS_DEFAULT if c in df.columns]
if not SEARCHABLE_FIELDS:
    # fallback: search everything
    SEARCHABLE_FIELDS = list(df.columns)

# Datasheet column (first candidate that exists)
DATASHEET_COL = next((c for c in DATASHEET_CANDIDATES if c in df.columns), None)

# Precompute normalized code and a lowercase "blob" for each row for fast search
def row_blob(row) -> str:
    parts = []
    for c in SEARCHABLE_FIELDS:
        parts.append(str(row.get(c, "")))
    blob = " ".join(parts).lower()
    # normalize whitespace and strip non-alnum except space
    blob = re.sub(r"[^a-z0-9\s]+", " ", blob)
    blob = re.sub(r"\s+", " ", blob).strip()
    return blob

df = df.copy()
df["__code_norm"] = df["Stock Code"].map(normalize_code)
df["__blob"] = df.apply(row_blob, axis=1)

# Quick exact lookup map by normalized code
code_index = {row["__code_norm"]: i for i, row in df.iterrows()}

# ---------- Endpoints ----------
@app.get("/items/by_code")
def by_code():
    """Exact Stock Code lookup (format-insensitive). ?q=CODE"""
    raw = (request.args.get("q") or "").strip()
    if not raw:
        return jsonify({"error": "Missing 'q' parameter"}), 400

    key = normalize_code(raw)
    idx = code_index.get(key)
    if idx is None:
        return jsonify({"error": f"No item found for code '{raw}'"}), 404

    item = df.iloc[idx].drop(labels=["__code_norm","__blob"]).to_dict()
    return jsonify({
        "query": raw,
        "match_type": "exact",
        "result": item
    })

@app.get("/items/search")
def search():
    """
    Keyword search across key fields.
    ?q=words&limit=10
    Returns ranked results with score and matched_terms.
    """
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"error": "Missing 'q' search parameter"}), 400
    if len(q) > 200:
        return jsonify({"error": "Query too long (max 200 chars)"}), 400

    limit = clamp(int(request.args.get("limit", 10)), 1, 25)
    terms = tokenize(q)
    if not terms:
        return jsonify({"error": "Your query had no meaningful terms after filtering."}), 400

    scored = []
    for i, row in df.iterrows():
        blob = row["__blob"]
        # score = total occurrences of all terms in blob
        score = 0
        matched_terms = []
        for t in terms:
            cnt = blob.count(t)
            if cnt > 0:
                score += cnt
                matched_terms.append(t)
        if score > 0:
            scored.append((score, matched_terms, i))

    if not scored:
        return jsonify({
            "query": q,
            "count": 0,
            "results": []
        }), 200

    scored.sort(key=lambda x: (-x[0], x[2]))  # score desc, original order tie-break
    top = scored[:limit]

    out = []
    for rank, (score, matched_terms, i) in enumerate(top, start=1):
        rec = df.iloc[i].drop(labels=["__code_norm","__blob"]).to_dict()
        out.append({
            "rank": rank,
            "score": score,
            "matched_terms": matched_terms,
            "record": rec
        })

    return jsonify({
        "query": q,
        "count": len(out),
        "results": out
    })

@app.get("/healthz")
def healthz():
    return jsonify({
        "ok": True,
        "rows": int(df.shape[0]),
        "columns": list(df.columns.drop(["__code_norm","__blob"])),
        "datasheet_column": DATASHEET_COL,
        "searchable_fields": SEARCHABLE_FIELDS
    })

if __name__ == "__main__":
    # For local dev; on Render use: gunicorn app:app
    app.run(host="0.0.0.0", port=8000, debug=True)
