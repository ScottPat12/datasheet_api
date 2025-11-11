from flask import Flask, request, jsonify
import pandas as pd
import os, re, datetime
from difflib import SequenceMatcher

app = Flask(__name__)

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = os.getenv("ITEMS_CSV_PATH", "data/Items.csv")
LOG_PATH = "logs/queries.csv"
FUZZY_THRESHOLD = 0.85
MAX_LIMIT = 25
DEFAULT_LIMIT = 10

# Common fields we expect (case-insensitive map added at load)
EXPECTED_FIELDS = [
    "Stock Code",
    "Name",
    "Stock Group",
    "Category",
    "Short description",
    "Marketing description",
    "Applications",
    "Benefits",
    "Data sheet",
]

STOP_WORDS = {
    "the", "and", "or", "of", "a", "an", "for", "to", "is", "are", "with", "on",
    "by", "from", "in", "at", "as"
}

# Minimal synonyms; extend as you like
SYNONYMS = {
    "halogen-free": ["halogen free", "lszh", "low smoke zero halogen"],
    "neoprene": ["pcp", "rubber"],
    "rubber": ["pcp", "neoprene"],
    "epr": ["rubber"],  # EPR insulation, often grouped colloquially as rubber
    "uv": ["sunlight", "sun-resistant", "uv-resistant"],
    "outdoor": ["external", "outside"],
    "underground": ["buried", "direct-burial", "direct burial"],
    "datasheet": ["data sheet", "spec sheet", "specification"],
}

# -----------------------------
# LOADING
# -----------------------------
try:
    df_items = pd.read_csv(DATA_PATH, dtype=str).fillna("")
except Exception as e:
    raise RuntimeError(f"Failed to load items file: {e}")

# Build a case-insensitive column resolver
COLMAP = {c.lower(): c for c in df_items.columns}

def C(name: str) -> str:
    """Return actual DF column name matched case-insensitively; fallback to input."""
    return COLMAP.get(name.lower(), name)

# Choose fields that are safe to concatenate for searching
SEARCHABLE_FIELDS = [
    f for f in [
        C("Stock Code"), C("Name"), C("Category"), C("Stock Group"),
        C("Short description"), C("Marketing description"),
        C("Applications"), C("Benefits")
    ] if f in df_items.columns
]

# Find datasheet column if present
DATASHEET_COL = C("Data sheet") if C("Data sheet") in df_items.columns else None

# -----------------------------
# NORMALIZATION
# -----------------------------
NON_WORDS = re.compile(r"[^a-z0-9]+")

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = NON_WORDS.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_code(s: str) -> str:
    # Stronger normalization for codes: remove all non-alnum and uppercase
    return re.sub(r"[^a-zA-Z0-9]", "", (s or "")).upper()

def expand_terms(tokens):
    expanded = set(tokens)
    for t in list(tokens):
        if t in SYNONYMS:
            for syn in SYNONYMS[t]:
                expanded.add(normalize_text(syn))
    return expanded

def tokenize(q: str):
    base = normalize_text(q)
    parts = [p for p in base.split() if p and p not in STOP_WORDS]
    return list(expand_terms(parts))

# -----------------------------
# PRECOMPUTE SEARCH BLOBS
# -----------------------------
def make_blob(row) -> str:
    parts = []
    for f in SEARCHABLE_FIELDS:
        parts.append(str(row.get(f, "")))
    return normalize_text(" ".join(parts))

df_items["__blob"] = df_items.apply(make_blob, axis=1)

# Also store a normalized code for exact/fuzzy matching
code_col = C("Stock Code") if C("Stock Code") in df_items.columns else None
if code_col:
    df_items["__code_norm"] = df_items[code_col].apply(normalize_code)
else:
    df_items["__code_norm"] = ""

# -----------------------------
# INTENT / SCORING / LOGGING
# -----------------------------
def classify_user_intent(query: str) -> str:
    q = (query or "").lower()
    if any(t in q for t in ["what is", "part number", "code "]): return "product_lookup"
    if any(t in q for t in ["voltage", "amp", "halogen", "rating"]): return "attribute_query"
    if any(t in q for t in ["alternative", "replace", "equivalent"]): return "substitute_search"
    if any(t in q for t in ["suitable for", "use for", "outdoor", "underground"]): return "application"
    if any(t in q for t in ["standard", "as/nzs", "compliant"]): return "regulation"
    return "general"

def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def calc_relevance(blob: str, terms) -> float:
    exact = 0.0
    fuzzy = 0.0
    for t in terms:
        if t in blob:
            exact += 2
        else:
            # basic word-based fuzzy
            best = max((fuzzy_ratio(t, w) for w in blob.split()), default=0.0)
            if best >= FUZZY_THRESHOLD:
                fuzzy += best
    return exact + fuzzy

def log_query(q, intent, count, route):
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.utcnow().isoformat()},{route},{intent},{count},{q}\n")
    except Exception as e:
        print("Log error:", e)

def clamp_limit(n: int) -> int:
    try:
        n = int(n)
    except:
        n = DEFAULT_LIMIT
    return max(1, min(MAX_LIMIT, n))

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/items/by_code")
def by_code():
    """Exact code match (case-insensitive, dash-insensitive)."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "Missing q"}), 400
    q_norm = normalize_code(q)
    matches = df_items[df_items["__code_norm"] == q_norm]
    if matches.empty:
        return jsonify({"error": f"No item for code {q}"}), 404
    return jsonify(matches.iloc[0].to_dict())

@app.route("/items/search")
def search_items():
    """Keyword search with scoring."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "Missing q"}), 400
    limit = clamp_limit(request.args.get("limit", DEFAULT_LIMIT))
    intent = classify_user_intent(q)

    terms = tokenize(q)
    results = []
    for _, row in df_items.iterrows():
        score = calc_relevance(row["__blob"], terms)
        if score > 0:
            results.append({"product": row.to_dict(), "score": round(score, 2)})

    results.sort(key=lambda r: -r["score"])
    results = results[:limit]

    confidence = round(min(0.4 + 0.1 * len(results), 0.95), 2)
    log_query(q, intent, len(results), route="/items/search")

    return jsonify({
        "query": q,
        "intent": intent,
        "confidence": confidence,
        "count": len(results),
        "results": [r["product"] for r in results]
    })

@app.route("/items/filter")
def filter_items():
    """
    Attribute-style filtering.
    Query params:
      - attributes: string of keywords (space/comma separated)
      - require: 'all' (default) or 'any'  -> require all tokens vs any token
      - exclude: keywords to exclude
      - category: optional category substring to require
      - limit: 1..25
      - offset: for pagination (default 0)
    """
    raw_attr = request.args.get("attributes", "").strip()
    if not raw_attr:
        return jsonify({"error": "Missing 'attributes'"}), 400

    require_mode = (request.args.get("require", "all").strip().lower() or "all")
    exclude_raw = request.args.get("exclude", "").strip()
    category_req = request.args.get("category", "").strip().lower()
    limit = clamp_limit(request.args.get("limit", DEFAULT_LIMIT))
    try:
        offset = max(0, int(request.args.get("offset", 0)))
    except:
        offset = 0

    # tokens
    req_terms = tokenize(raw_attr)
    exc_terms = tokenize(exclude_raw) if exclude_raw else []

    def row_matches(row) -> (bool, set):
        blob = row["__blob"]
        # category filter
        if category_req and C("Category") in row and category_req not in row.get(C("Category"), "").lower():
            return False, set()

        present = {t for t in req_terms if t in blob}
        # require all vs any
        if require_mode == "all" and len(present) < len(req_terms):
            return False, present
        if require_mode == "any" and len(present) == 0:
            return False, present

        # exclusions
        for x in exc_terms:
            if x in blob:
                return False, present
        return True, present

    matches = []
    for _, row in df_items.iterrows():
        ok, matched_terms = row_matches(row)
        if ok:
            matches.append({
                "product": row.to_dict(),
                "matched_terms": sorted(list(matched_terms))
            })

    total = len(matches)
    matches = matches[offset: offset + limit]

    # simple confidence
    confidence = round(min(0.4 + 0.1 * len(matches), 0.95), 2)
    intent = "filter"
    log_query(raw_attr, intent, total, route="/items/filter")

    return jsonify({
        "query": raw_attr,
        "intent": intent,
        "require": require_mode,
        "confidence": confidence,
        "count": len(matches),
        "total": total,
        "offset": offset,
        "results": [m["product"] for m in matches]
    })

@app.route("/ask", methods=["POST"])
def ask_semantic():
    """
    Placeholder for Phase 2 RAG.
    For now, proxies to /items/search.
    Body: { "query": "...", "limit": 5 }
    """
    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    limit = clamp_limit(data.get("limit", DEFAULT_LIMIT))
    if not query:
        return jsonify({"error": "No query provided"}), 400
    with app.test_request_context(f"/items/search?q={query}&limit={limit}"):
        return search_items()

@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "records": int(df_items.shape[0])})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
