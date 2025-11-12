from flask import Flask, request, jsonify
import pandas as pd, os, datetime, re
from difflib import SequenceMatcher

app = Flask(__name__)

# === CONFIG ======================================================
DATA_PATH = os.getenv("ITEMS_CSV_PATH", "data/Items.csv")
API_KEY = os.getenv("API_KEY", None)
LOG_PATH = "logs/queries.csv"

# fuzzy thresholds
FUZZY_THRESHOLD = 0.85          # for general fuzzy token match
CODE_FUZZY_THRESHOLD = 0.80     # slightly more lenient for code aliases

# === NORMALIZATION HELPERS =======================================
_SEP_RE = re.compile(r"[ \-\_/]")
_MM2_RE = re.compile(r"(mm²|mm2|sqmm)", re.IGNORECASE)

def norm_mm2(text: str) -> str:
    # normalize mm² / mm2 / sqmm → mm2
    return _MM2_RE.sub("mm2", text)

def normalize_code(text: str) -> str:
    """
    Aggressive code normalization so these all align:
    MC05/2.5, MC05 2.5, MC05-2.5, mc05 2.5mm²  → mc052p5 (and lowercased)
    Rules:
      - lowercase
      - normalize mm² tokens to mm2
      - remove spaces, hyphens, underscores, slashes
      - replace decimal '.' with 'p' (so 2.5 != 25)
    """
    if not isinstance(text, str):
        text = str(text or "")
    t = text.strip().lower()
    t = norm_mm2(t)
    t = t.replace(".", "p")
    t = _SEP_RE.sub("", t)
    return t

def code_aliases(raw_code: str):
    """
    Return a small set of aliases for a stock code to catch common human variants.
    """
    raw = (raw_code or "").strip()
    low = raw.lower()
    no_space = low.replace(" ", "")
    no_hyphen = low.replace("-", "")
    no_slash = low.replace("/", "")
    dotted_p = low.replace(".", "p")
    normalized = normalize_code(raw)

    # de-dup while preserving order
    seen, aliases = set(), []
    for cand in [low, no_space, no_hyphen, no_slash, dotted_p, normalized]:
        if cand and cand not in seen:
            aliases.append(cand); seen.add(cand)
    return aliases

def tokenize(query: str):
    return [t for t in re.split(r"[\s,;]+", (query or "").strip()) if t]

# === LOAD DATA ===================================================
try:
    df_items = pd.read_csv(DATA_PATH, dtype=str).fillna("")
except Exception as e:
    raise RuntimeError(f"Failed to load items file: {e}")

# compute code aliases + normalized columns
if "Stock Code" not in df_items.columns:
    # fail fast with a helpful message
    raise RuntimeError("Items.csv must include a 'Stock Code' column.")

df_items["__code_raw"]  = df_items["Stock Code"].str.strip().str.lower()
df_items["__code_norm"] = df_items["Stock Code"].apply(normalize_code)
df_items["__code_aliases"] = df_items["Stock Code"].apply(code_aliases)

# Pre-compute search blobs (include aliases to boost discoverability)
def make_search_blob(row):
    base = " ".join(str(v).lower() for v in row.values)
    ali  = " ".join(row["__code_aliases"])
    return f"{base} {ali}"

df_items["__search_blob"] = df_items.apply(make_search_blob, axis=1)

# === INTENT & SCORING ===========================================
def classify_user_intent(query: str) -> str:
    q = (query or "").lower()
    if any(t in q for t in ["what is", "part number", "code "]): return "product_lookup"
    if any(t in q for t in ["voltage", "amp", "halogen", "rating"]): return "attribute_query"
    if any(t in q for t in ["alternative", "replace", "equivalent"]): return "substitute_search"
    if any(t in q for t in ["suitable for", "use for", "outdoor", "underground"]): return "application"
    if any(t in q for t in ["standard", "as/nzs", "compliant"]): return "regulation"
    return "general"

def best_similarity(token: str, words_iter):
    best = 0.0
    for w in words_iter:
        s = SequenceMatcher(None, token, w).ratio()
        if s > best:
            best = s
            if best >= 0.999:
                break
    return best

def calc_relevance(row, terms):
    """
    Combined exact + fuzzy score.
    Heavy boost if term matches code aliases (esp normalized forms).
    """
    blob = row["__search_blob"]
    words = blob.split()
    aliases = row["__code_aliases"]

    score = 0.0
    for t in terms:
        tl = t.lower()
        tn = normalize_code(t)

        # 1) Code alias matches (highest weight)
        if tl in aliases or tn in aliases:
            score += 6.0
            continue
        # fuzzy on aliases
        sim_alias = max(best_similarity(tl, aliases), best_similarity(tn, aliases))
        if sim_alias >= CODE_FUZZY_THRESHOLD:
            score += 3.5
            continue

        # 2) General exact match in blob
        if tl in blob:
            score += 2.0
            continue

        # 3) General fuzzy match in blob
        sim = best_similarity(tl, words)
        if sim >= FUZZY_THRESHOLD:
            score += sim  # fractional bump

    return round(score, 3)

def log_query(q, intent, results_count, extras=None):
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.utcnow()},{q},{intent},{results_count},{extras or ''}\n")
    except Exception as e:
        print("Log error:", e)

def compute_confidence(results):
    # better than simple count-only: blend quantity + average score + top consistency
    if not results:
        return 0.0
    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores)
    top = scores[:3] if len(scores) >= 3 else scores
    consistency = 1.0
    if len(top) > 1:
        spread = max(top) - min(top)
        consistency = max(0.0, 1.0 - spread / max(top))
    qty = min(len(results) / 5.0, 1.0)
    conf = 0.45 * min(avg / 8.0, 1.0) + 0.25 * consistency + 0.30 * qty
    return round(min(max(conf, 0.0), 0.95), 2)

# === SECURITY ====================================================
@app.before_request
def require_api_key():
    # You said you’re not using API key right now; leave hook but allow all.
    if False and request.path not in ("/", "/healthz"):
        if API_KEY and request.headers.get("X-API-Key") != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

# === ROUTES ======================================================
@app.route("/items/search")
def search_items():
    q = request.args.get("q", "").strip()
    limit = int(request.args.get("limit", 10))
    if not q:
        return jsonify({"error": "Missing q"}), 400

    intent = classify_user_intent(q)
    terms = tokenize(q)

    # also push normalized code-like tokens so "MC05/2.5" aligns with "MC05 2.5"
    terms_norm = list({normalize_code(t) for t in terms if t})
    terms_all = terms + terms_norm

    results = []
    for _, row in df_items.iterrows():
        s = calc_relevance(row, terms_all)
        if s > 0:
            results.append({"product": row.to_dict(), "score": s})

    results.sort(key=lambda r: -r["score"])
    results = results[:limit]
    conf = compute_confidence(results)

    log_query(q, intent, len(results), extras=f"conf={conf}")
    return jsonify({
        "query": q, "intent": intent,
        "confidence": conf,
        "count": len(results),
        "results": [r["product"] for r in results]
    })

@app.route("/items/by_code")
def by_code():
    """
    Robust code lookup:
      1) case-insensitive exact 'Stock Code'
      2) normalized match against __code_norm
      3) fuzzy against __code_aliases (suggestions if multiple)
    """
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "Missing q"}), 400

    q_low = q.lower()
    q_norm = normalize_code(q)

    # 1) exact case-insensitive match
    exact = df_items[df_items["__code_raw"] == q_low]
    if not exact.empty:
        return jsonify(exact.iloc[0].to_dict())

    # 2) normalized match
    normed = df_items[df_items["__code_norm"] == q_norm]
    if not normed.empty:
        return jsonify(normed.iloc[0].to_dict())

    # 3) fuzzy over aliases → collect candidates
    candidates = []
    for _, row in df_items.iterrows():
        aliases = row["__code_aliases"]
        sim = max(best_similarity(q_low, aliases), best_similarity(q_norm, aliases))
        if sim >= CODE_FUZZY_THRESHOLD:
            candidates.append((sim, row.to_dict()))
    if len(candidates) == 1:
        return jsonify(candidates[0][1])
    elif len(candidates) > 1:
        # return top 5 suggestions
        candidates.sort(key=lambda x: -x[0])
        sugg = [{"similarity": round(s, 3), "item": d} for s, d in candidates[:5]]
        return jsonify({"error": f"No exact item for code '{q}'", "suggestions": sugg}), 404

    # nothing found
    return jsonify({"error": f"No item for code '{q}'"}), 404

@app.route("/ask", methods=["POST"])
def ask_semantic():
    """
    Phase 2 placeholder – currently proxies to /items/search.
    """
    data = request.get_json(force=True)
    query = data.get("query", "")
    limit = data.get("limit", 5)
    with app.test_request_context(f"/items/search?q={query}&limit={limit}"):
        return search_items()

@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "records": len(df_items)})

# === RUN =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
