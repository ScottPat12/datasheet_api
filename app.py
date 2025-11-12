from flask import Flask, request, jsonify
import pandas as pd, os, re, datetime
from rapidfuzz import fuzz

app = Flask(__name__)

# ======================================================
# CONFIG
# ======================================================
DATA_PATH = os.getenv("ITEMS_CSV_PATH", "data/Items.csv")
LOG_PATH = "logs/queries.csv"
FUZZY_THRESHOLD = 80  # 0–100, higher = stricter match

# ======================================================
# LOAD DATA
# ======================================================
try:
    df_items = pd.read_csv(DATA_PATH, dtype=str).fillna("")
except Exception as e:
    raise RuntimeError(f"Failed to load items file: {e}")

# Precompute lowercase blob for full-text matching
def make_blob(row):
    return " ".join(str(v).lower() for v in row.values)

df_items["__blob"] = df_items.apply(make_blob, axis=1)

# ======================================================
# HELPERS
# ======================================================
def log_query(q, endpoint, n):
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.utcnow()},{endpoint},{q},{n}\n")
    except Exception:
        pass

def numeric_filter(df, field, op, val):
    """Try numeric compare if column exists and values are numeric."""
    if field not in df.columns:
        return df
    try:
        series = pd.to_numeric(df[field], errors="coerce")
        val = float(val)
        if op == ">":  return df[series > val]
        if op == ">=": return df[series >= val]
        if op == "<":  return df[series < val]
        if op == "<=": return df[series <= val]
        if op == "=":  return df[series == val]
    except Exception:
        return df
    return df

def text_match_score(blob, terms):
    score = 0
    for t in terms:
        if t in blob:
            score += 2
        else:
            sim = max(fuzz.partial_ratio(t, w) for w in blob.split())
            if sim > FUZZY_THRESHOLD:
                score += sim / 100
    return score

# ======================================================
# ROUTES
# ======================================================

@app.route("/healthz")
def health():
    return jsonify({"ok": True, "records": len(df_items)})

# --- Exact Code Lookup ------------------------------------------------
@app.route("/items/by_code")
def by_code():
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify({"error": "Missing q"}), 400
    matches = df_items[df_items["Stock Code"].str.lower() == q]
    if matches.empty:
        # try fuzzy suggestions
        df_items["score"] = df_items["Stock Code"].apply(lambda c: fuzz.ratio(q, c.lower()))
        suggestions = (
            df_items.sort_values("score", ascending=False)
            .head(3)["Stock Code"]
            .tolist()
        )
        return jsonify({"error": f"No item for code '{q}'", "suggestions": suggestions}), 404
    return jsonify(matches.iloc[0].to_dict())

# --- Keyword Search ---------------------------------------------------
@app.route("/items/search")
def search_items():
    q = request.args.get("q", "").strip()
    limit = int(request.args.get("limit", 10))
    if not q:
        return jsonify({"error": "Missing q"}), 400

    terms = [t.lower() for t in re.split(r"[\s,;]+", q) if t]
    results = []
    for _, row in df_items.iterrows():
        score = text_match_score(row["__blob"], terms)
        if score > 0:
            results.append({"product": row.to_dict(), "score": round(score, 2)})
    results.sort(key=lambda r: -r["score"])
    top = results[:limit]
    log_query(q, "search", len(top))
    return jsonify({
        "query": q,
        "count": len(top),
        "results": [r["product"] for r in top]
    })

# --- Filtered Search (Advanced) ---------------------------------------
@app.route("/items/filter")
def filter_items():
    """Support structured numeric and keyword filters."""
    df = df_items.copy()
    # Example: ?voltage>=450&core_size=10&include=flexible,halogen
    for k, v in request.args.items():
        if k.lower().startswith("voltage"):
            op = re.findall(r"[><=]+", k)
            df = numeric_filter(df, "Voltage", op[0] if op else "=", v)
        elif k.lower().startswith("core") and "size" in k.lower():
            df = numeric_filter(df, "Core Size", "=", v)
        elif k.lower() in ("include", "require"):
            includes = [x.strip().lower() for x in v.split(",")]
            df = df[df["__blob"].apply(lambda b: all(t in b for t in includes))]
        elif k.lower() in ("exclude", "not"):
            excludes = [x.strip().lower() for x in v.split(",")]
            df = df[~df["__blob"].apply(lambda b: any(t in b for t in excludes))]

    limit = int(request.args.get("limit", 20))
    sample = df.head(limit)
    log_query(str(request.args), "filter", len(sample))
    return jsonify({
        "query": dict(request.args),
        "count": len(sample),
        "results": sample.to_dict(orient="records")
    })

# --- Compare Two Codes ------------------------------------------------
@app.route("/items/compare")
def compare_items():
    """Compare two product codes and show differences."""
    codes = request.args.getlist("code")
    if len(codes) < 2:
        return jsonify({"error": "Need ?code=A&code=B"}), 400
    subset = df_items[df_items["Stock Code"].isin(codes)]
    if subset.empty:
        return jsonify({"error": "No matching codes"}), 404
    # Align on columns
    records = subset.to_dict(orient="records")
    diffs = {}
    cols = df_items.columns
    if len(records) == 2:
        a, b = records
        for c in cols:
            if a[c] != b[c]:
                diffs[c] = {"A": a[c], "B": b[c]}
    return jsonify({"codes": codes, "diffs": diffs})

# --- Ask Endpoint (placeholder for GPT / RAG later) -------------------
@app.route("/ask", methods=["POST"])
def ask_semantic():
    data = request.get_json(force=True)
    query = data.get("query", "")
    limit = data.get("limit", 5)
    with app.test_request_context(f"/items/search?q={query}&limit={limit}"):
        return search_items()

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
