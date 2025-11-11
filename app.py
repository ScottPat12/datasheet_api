from flask import Flask, request, jsonify
import pandas as pd, os, datetime, re
from difflib import SequenceMatcher

app = Flask(__name__)

# === CONFIG ======================================================
DATA_PATH = os.getenv("ITEMS_CSV_PATH", "data/Items.csv")
API_KEY = os.getenv("API_KEY", None)
LOG_PATH = "logs/queries.csv"
FUZZY_THRESHOLD = 0.85

# === LOAD DATA ===================================================
try:
    df_items = pd.read_csv(DATA_PATH, dtype=str).fillna("")
except Exception as e:
    raise RuntimeError(f"Failed to load items file: {e}")

# Pre-compute search blobs
def make_search_blob(row):
    return " ".join(str(v).lower() for v in row.values())

df_items["__search_blob"] = df_items.apply(make_search_blob, axis=1)

# === HELPER FUNCTIONS ============================================
def classify_user_intent(query: str) -> str:
    q = query.lower()
    if any(t in q for t in ["what is", "part number", "code "]): return "product_lookup"
    if any(t in q for t in ["voltage", "amp", "halogen", "rating"]): return "attribute_query"
    if any(t in q for t in ["alternative", "replace", "equivalent"]): return "substitute_search"
    if any(t in q for t in ["suitable for", "use for", "outdoor", "underground"]): return "application"
    if any(t in q for t in ["standard", "as/nzs", "compliant"]): return "regulation"
    return "general"

def calc_relevance(blob: str, terms):
    exact = fuzzy = 0.0
    for t in terms:
        if t in blob: exact += 2
        else:
            best = max((SequenceMatcher(None, t, w).ratio() for w in blob.split()), default=0)
            if best >= FUZZY_THRESHOLD: fuzzy += best
    return exact + fuzzy

def log_query(q, intent, results):
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.utcnow()},{q},{intent},{len(results)}\n")
    except Exception as e:
        print("Log error:", e)

# === SECURITY ====================================================
@app.before_request
def require_api_key():
    if request.path not in ("/", "/healthz"):
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
    terms = [t.lower() for t in re.split(r"[\s,;]+", q) if t]
    results = []
    for _, row in df_items.iterrows():
        score = calc_relevance(row["__search_blob"], terms)
        if score > 0:
            results.append({"product": row.to_dict(), "score": round(score, 2)})
    results.sort(key=lambda r: -r["score"])
    results = results[:limit]
    log_query(q, intent, results)
    confidence = min(0.4 + 0.1 * len(results), 0.9)
    return jsonify({
        "query": q, "intent": intent,
        "confidence": round(confidence, 2),
        "count": len(results),
        "results": [r["product"] for r in results]
    })

@app.route("/items/by_code")
def by_code():
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify({"error": "Missing q"}), 400
    match = df_items[df_items["Stock Code"].str.lower() == q]
    if match.empty:
        return jsonify({"error": f"No item for code {q}"}), 404
    return jsonify(match.iloc[0].to_dict())

@app.route("/ask", methods=["POST"])
def ask_semantic():
    """
    Phase 2 placeholder – will use vector index later.
    Currently proxies to /items/search.
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
    app.run(host="0.0.0.0", port=8000)
