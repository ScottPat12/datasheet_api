# app.py
import os, re, json, time, uuid, math, io
from difflib import SequenceMatcher
from collections import Counter, defaultdict

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

# ----------------------------
# Config / Environment
# ----------------------------
ITEMS_CSV_PATH = os.getenv("ITEMS_CSV_PATH", "data/Items.csv")
SERIES_CSV_PATH = os.getenv("SERIES_CSV_PATH", "data/Series.csv")  # optional
DIRECT_T = float(os.getenv("DIRECT_T", 0.80))
LIKELY_T = float(os.getenv("LIKELY_T", 0.55))
API_KEY = os.getenv("API_KEY", "")  # optional; if empty, auth disabled
APP_VERSION = os.getenv("APP_VERSION", "v1")

# generic scoring knobs
PHRASE_MAX_N = 4               # check 2..4-grams
PHRASE_BONUS = 1.0             # added once per matched phrase
PROX_WINDOW = 3                # terms within +/-3 tokens
PROX_BONUS = 0.25              # added per close pair
FUZZY_TERM_PENALTY = 0.15
FIELD_WEIGHTS = {
    # generic weights — no domain terms
    "Stock Code": 2.5,
    "Stock Group": 2.0,
    "Name": 1.8,
    "Category": 1.4,
    # series fields (from Series.csv)
    "Series name": 2.2,
    "Short description": 1.6,
    "Marketing description": 1.4,
}

# ----------------------------
# App init
# ----------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

_last_data_load_at = None

# ----------------------------
# Helpers
# ----------------------------
TOKEN_SPLIT_RE = re.compile(r"[^\w\.]+", re.UNICODE)

def normalize_code(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    # collapse spaces, dashes, underscores, slashes; keep alnum and dots
    s = re.sub(r"[\s\-_/]+", "", s)
    return s

def code_prefix(s: str) -> str:
    s = (s or "").strip().lower()
    m = re.match(r"([a-z]+)", re.sub(r"[^a-z0-9]", "", s))
    return m.group(1) if m else ""

def tokenize_text(text: str):
    return [t for t in TOKEN_SPLIT_RE.split((text or "").lower()) if t]

def ngrams(tokens, n):
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i:i+n])

def parse_mm2(s):
    s = (s or "").replace(",", "").lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mm2|mm²|sqmm)?", s)
    return float(m.group(1)) if m else None

def parse_volts(s):
    s = (s or "").replace(" ", "").replace(",", "").lower()
    m = re.search(r"(\d+(?:\.\d+)?)(kv|v)", s)
    if not m: return None
    val = float(m.group(1)); unit = m.group(2)
    return val * 1000.0 if unit == "kv" else val

# ----------------------------
# Load Series table (optional)
# ----------------------------
def try_load_series(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype=str).fillna("")
    # normalise headers (case-insensitive mapping)
    colmap = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("stock group", "stock_group"):
            colmap[c] = "Stock Group"
        elif cl in ("series name", "series", "series_name"):
            colmap[c] = "Series name"
        elif cl in ("short description", "short_desc", "short"):
            colmap[c] = "Short description"
        elif cl in ("marketing description", "marketing_desc", "marketing"):
            colmap[c] = "Marketing description"
    if colmap:
        df = df.rename(columns=colmap)
    # keep only known columns
    keep = [c for c in ["Stock Group","Series name","Short description","Marketing description"] if c in df.columns]
    if not keep or "Stock Group" not in keep:
        return None
    return df[keep].copy()

# ----------------------------
# Data loading + corpus build
# ----------------------------
def load_items(items_csv: str, series_csv: str | None) -> tuple[pd.DataFrame, dict]:
    global _last_data_load_at

    items = pd.read_csv(items_csv, dtype=str).fillna("")
    if "Stock Code" not in items.columns:
        raise ValueError("Items.csv must include a 'Stock Code' column")
    # ensure Stock Group present (can be blank)
    if "Stock Group" not in items.columns:
        items["Stock Group"] = ""

    # optional series merge
    s_df = try_load_series(series_csv) if series_csv else None
    if s_df is not None:
        # deduplicate series by Stock Group (keep first)
        s_df = s_df.drop_duplicates(subset=["Stock Group"])
        items = items.merge(s_df, on="Stock Group", how="left")
    else:
        # ensure series columns exist (blank) so code stays simple
        for c in ["Series name","Short description","Marketing description"]:
            if c not in items.columns: items[c] = ""

    # precompute code norms
    items["__norm_code"] = items["Stock Code"].map(normalize_code)
    items["__code_prefix"] = items["Stock Code"].map(code_prefix)

    # lowercase copies for all visible columns
    field_texts = {}
    for col in items.columns:
        if col.startswith("__"):
            continue
        field_texts[col] = items[col].astype(str).str.lower().fillna("")

    # assemble per-row blobs & per-field tokens
    unified = []
    tokens = []
    per_field_tokens = []
    for i in range(len(items)):
        parts = []
        for col in items.columns:
            if col.startswith("__"):
                continue
            parts.append(field_texts[col].iat[i])
        blob = " ".join(parts)
        unified.append(blob)
        toks = tokenize_text(blob)
        tokens.append(toks)

        f_toks = {}
        for col, series in field_texts.items():
            f_toks[col] = tokenize_text(series.iat[i])
        per_field_tokens.append(f_toks)

    items["__search_blob"] = unified
    items["__tokens"] = tokens
    items["__field_tokens"] = per_field_tokens

    # IDF on corpus
    items["_tmp_all_tokens"] = items["__tokens"].apply(set)
    docfreq = Counter()
    for s in items["_tmp_all_tokens"]:
        docfreq.update(s)
    N = len(items)
    idf = {}
    for term, dfreq in docfreq.items():
        idf[term] = math.log((1 + N) / (1 + dfreq)) + 1.0
    items.drop(columns=["_tmp_all_tokens"], inplace=True)

    _last_data_load_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return items, idf

try:
    _df, _idf = load_items(ITEMS_CSV_PATH, SERIES_CSV_PATH)
except Exception as e:
    _df = pd.DataFrame()
    _idf = {}
    _data_load_error = str(e)
else:
    _data_load_error = ""

# ----------------------------
# Logging (stdout JSONL)
# ----------------------------
def log_event(payload: dict):
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "request_id": str(uuid.uuid4()),
        "version": APP_VERSION,
        **payload,
    }
    print(json.dumps(payload), flush=True)

# ----------------------------
# Security helper
# ----------------------------
def require_key():
    if not API_KEY:
        return True
    from hmac import compare_digest
    return compare_digest(request.headers.get("X-API-Key", ""), API_KEY)

# ----------------------------
# Numeric filters (generic)
# ----------------------------
def apply_numeric_filters(row: dict, ops: dict):
    reasons = []
    size = parse_mm2(row.get("Size") or row.get("Core Size") or "")
    volt = parse_volts(row.get("Voltage") or row.get("Max Voltage") or "")

    def check(val, op, target, label):
        tgt = float(target)
        if val is None:
            reasons.append(f"{label}:missing")
            return False
        if op.endswith("__gte") and val < tgt:
            reasons.append(f"{label}<{tgt}")
            return False
        if op.endswith("__lte") and val > tgt:
            reasons.append(f"{label}>{tgt}")
            return False
        return True

    ok = True
    if "size__gte" in ops: ok &= check(size, "size__gte", ops["size__gte"], "size")
    if "size__lte" in ops: ok &= check(size, "size__lte", ops["size__lte"], "size")
    if "voltage__gte" in ops: ok &= check(volt, "voltage__gte", ops["voltage__gte"], "voltage")
    if "voltage__lte" in ops: ok &= check(volt, "voltage__lte", ops["voltage__lte"], "voltage")

    return ok, reasons

# ----------------------------
# Generic scoring (no domain hard-codes)
# ----------------------------
def tokenize_query(q: str):
    q = (q or "").strip().lower()
    toks = [t for t in TOKEN_SPLIT_RE.split(q) if t]
    toks = [t for t in toks if len(t) > 1]  # drop 1-char noise
    return toks

def tfidf_score(row_tokens, query_tokens):
    tf = Counter(row_tokens)
    num = 0.0
    den_row = 0.0
    den_q = 0.0
    q_vec = {}
    for t in query_tokens:
        w = _idf.get(t, 1.0)
        q_vec[t] = q_vec.get(t, 0.0) + w
    for t, q_w in q_vec.items():
        d_w = (tf.get(t, 0) * _idf.get(t, 1.0))
        num += q_w * d_w
        den_row += d_w * d_w
        den_q += q_w * q_w
    if den_row == 0.0 or den_q == 0.0:
        return 0.0
    return num / math.sqrt(den_row * den_q)

def phrase_bonus(row_blob: str, q_tokens):
    bonus = 0.0
    if len(q_tokens) < 2:
        return 0.0
    rb = row_blob
    for n in range(2, min(PHRASE_MAX_N, len(q_tokens)) + 1):
        for p in ngrams(q_tokens, n):
            if p in rb:
                bonus += PHRASE_BONUS
    return bonus

def proximity_bonus(row_tokens, q_tokens):
    uniq = list(dict.fromkeys(q_tokens))
    pos = defaultdict(list)
    for i, tok in enumerate(row_tokens):
        pos[tok].append(i)
    bonus = 0.0
    for i in range(len(uniq)):
        for j in range(i+1, len(uniq)):
            a, b = uniq[i], uniq[j]
            if not pos[a] or not pos[b]:
                continue
            md = min(abs(pa - pb) for pa in pos[a] for pb in pos[b])
            if md <= PROX_WINDOW:
                bonus += PROX_BONUS
    return bonus

def field_weight_bonus(field_tokens_map, q_tokens):
    bonus = 0.0
    qset = set(q_tokens)
    for field, toks in field_tokens_map.items():
        if not toks:
            continue
        w = FIELD_WEIGHTS.get(field)
        if w and qset.intersection(toks):
            bonus += w * 0.15  # small additive bump per weighted field present
    return bonus

def fuzzy_term_adjust(row_blob: str, term: str) -> float:
    if term in row_blob:
        return 0.0
    s = SequenceMatcher(None, term, row_blob).ratio()
    return max(0.0, s - FUZZY_TERM_PENALTY)

def generic_score(row, q_tokens):
    base = tfidf_score(row["__tokens"], q_tokens)
    p_bonus = phrase_bonus(row["__search_blob"], q_tokens)
    prox = proximity_bonus(row["__tokens"], q_tokens)
    f_bonus = field_weight_bonus(row["__field_tokens"], q_tokens)
    fz = sum(fuzzy_term_adjust(row["__search_blob"], t) for t in q_tokens)
    return base + p_bonus + prox + f_bonus + 0.1 * fz

# ----------------------------
# Response helpers
# ----------------------------
def df_row_to_item(row) -> dict:
    return {c: row[c] for c in _df.columns if not c.startswith("__")}

def suggest_near_codes(norm_q: str, k=3):
    if _df.empty:
        return []
    sims = [(SequenceMatcher(None, norm_q, c).ratio(), i) for i, c in enumerate(_df["__norm_code"].tolist())]
    sims.sort(reverse=True)
    out = []
    for score, idx in sims[:k]:
        item = df_row_to_item(_df.iloc[idx])
        out.append({"score": round(float(score), 3), "item": item})
    return out

def classify_user_intent(q: str) -> str:
    ql = (q or "").lower()
    if re.search(r"[a-z]\d|/", ql):
        return "product_lookup"
    if any(k in ql for k in ["mm2", "mm²", "kv", "voltage", "outdoor", "armour", "armored", "armoured", "screen"]):
        return "attribute_query"
    if any(k in ql for k in ["regulation", "standard", "as/nzs", "code of practice"]):
        return "regulation"
    if any(k in ql for k in ["substitute", "alternative", "equivalent"]):
        return "substitute_search"
    return "general"

# ----------------------------
# Endpoints
# ----------------------------
@app.route("/healthz", methods=["GET"])
def healthz():
    if _data_load_error:
        return jsonify({"ok": False, "error": _data_load_error}), 500
    return jsonify({
        "ok": True,
        "records": int(len(_df.index)),
        "version": APP_VERSION,
        "last_data_load_at": _last_data_load_at,
        "series_merge": bool("Series name" in _df.columns and _df["Series name"].replace("", pd.NA).notna().any())
    })

@app.route("/items/by_code", methods=["GET"])
def by_code():
    if not require_key():
        return jsonify({"error": "unauthorized"}), 401
    q = request.args.get("q", "")
    if not q:
        return jsonify({"error": "missing q"}), 400
    n = normalize_code(q)
    res = _df[_df["__norm_code"] == n]
    if len(res) == 1:
        item = df_row_to_item(res.iloc[0])
        log_event({"endpoint": "/items/by_code", "query": q, "intent": "product_lookup", "decision": {"hit": True}})
        return jsonify(item)
    suggestions = suggest_near_codes(n, k=3)
    log_event({"endpoint": "/items/by_code", "query": q, "intent": "product_lookup", "decision": {"hit": False, "suggestions": len(suggestions)}})
    return jsonify({"error": "not_found", "message": "No exact code match", "suggestions": suggestions}), 404

@app.route("/items/search", methods=["GET"])
def search_items():
    if not require_key():
        return jsonify({"error": "unauthorized"}), 401

    q = request.args.get("q", "")
    limit = max(1, min(int(request.args.get("limit", "5")), 10))
    q_tokens = tokenize_query(q)
    intent = classify_user_intent(q)

    results = []
    if not _df.empty and q_tokens:
        for idx, row in _df.iterrows():
            sc = generic_score(row, q_tokens)
            if sc > 0:
                results.append({"_score": sc, "_idx": idx})

    results.sort(key=lambda r: r["_score"], reverse=True)
    top = results[0]["_score"] if results else 0.0
    # soft normalization to 0..1 (keeps tiers intuitive)
    confidence = max(0.0, min(1.0, float(top) / (1.0 + PHRASE_BONUS)))
    tier = "direct" if confidence >= DIRECT_T else "likely" if confidence >= LIKELY_T else "guidance"

    out = [df_row_to_item(_df.iloc[r["_idx"]]) | {"_score": round(float(r["_score"]), 3)} for r in results[:limit]]

    resp = {
        "query": q,
        "intent": intent,
        "tier": tier,
        "confidence": round(confidence, 3),
        "count": len(out),
        "results": out
    }
    log_event({"endpoint": "/items/search", "query": q, "intent": intent,
               "decision": {"tier": tier, "confidence": confidence, "returned": len(out)}})
    return jsonify(resp)

@app.route("/items/filter", methods=["GET", "POST"])
def filter_items():
    if not require_key():
        return jsonify({"error": "unauthorized"}), 401

    include = []
    exclude = []
    numeric_ops = {}
    limit = 10

    if request.method == "POST" and request.is_json:
        body = request.get_json(silent=True) or {}
        include = body.get("include", []) or []
        exclude = body.get("exclude", []) or []
        numeric_ops = body.get("numeric", {}) or {}
        limit = int(body.get("limit", 10))
    else:
        inc = request.args.getlist("include") or request.args.get("include", "")
        exc = request.args.getlist("exclude") or request.args.get("exclude", "")
        if isinstance(inc, list) and len(inc) == 1: inc = inc[0]
        if isinstance(exc, list) and len(exc) == 1: exc = exc[0]
        include = [t.strip().lower() for t in (inc.split(",") if isinstance(inc, str) else inc) if t.strip()]
        exclude = [t.strip().lower() for t in (exc.split(",") if isinstance(exc, str) else exc) if t.strip()]
        limit = int(request.args.get("limit", 10))
        for k in ("size__gte", "size__lte", "voltage__gte", "voltage__lte"):
            if k in request.args:
                numeric_ops[k] = request.args.get(k)

    limit = max(1, min(limit, 25))

    def row_passes(row):
        blob = row["__search_blob"]
        for t in include:
            if t not in blob:
                return False, [f"missing:{t}"]
        for t in exclude:
            if t in blob:
                return False, [f"excluded:{t}"]
        ok, reasons = apply_numeric_filters({c: row[c] for c in _df.columns if not c.startswith("__")}, numeric_ops)
        return ok, reasons

    candidates = len(_df.index)
    passed, why_out_collected = [], []

    for idx, row in _df.iterrows():
        ok, reasons = row_passes(row)
        if ok:
            passed.append(idx)
        else:
            if len(why_out_collected) < 10 and reasons:
                why_out_collected.append(";".join(reasons))

    out_rows = passed[:limit]
    out = [df_row_to_item(_df.iloc[i]) for i in out_rows]

    summary = {
        "include": include,
        "exclude": exclude,
        "numeric": numeric_ops,
        "candidates": candidates,
        "returned": len(out),
        "why_out_sample": why_out_collected[:5]
    }

    log_event({"endpoint": "/items/filter",
               "query": "N/A",
               "intent": "attribute_query",
               "filters": summary,
               "decision": {"tier": "direct" if len(out) else "guidance",
                            "confidence": 0.8 if len(out) else 0.3}})

    return jsonify({"results": out, "filter_summary": summary})

@app.route("/ask", methods=["POST"])
def ask():
    if not require_key():
        return jsonify({"error": "unauthorized"}), 401

    body = request.get_json(silent=True) or {}
    q = (body.get("query") or "").strip()
    limit = max(1, min(int(body.get("limit", 5)), 10))
    if not q:
        return jsonify({"error": "missing query"}), 400

    intent = classify_user_intent(q)
    q_tokens = tokenize_query(q)

    # product lookup: exact code first
    if intent == "product_lookup":
        n = normalize_code(q)
        res = _df[_df["__norm_code"] == n]
        if len(res) == 1:
            item = df_row_to_item(res.iloc[0])
            log_event({"endpoint": "/ask", "query": q, "intent": intent, "decision": {"route": "by_code", "hit": True}})
            return jsonify({"tier": "direct", "confidence": 0.95, "results": [item]})

    # generic ranked search
    ranked = []
    for idx, row in _df.iterrows():
        sc = generic_score(row, q_tokens)
        if sc > 0:
            ranked.append({"_score": sc, "_idx": idx})
    ranked.sort(key=lambda r: r["_score"], reverse=True)
    out = [df_row_to_item(_df.iloc[r["_idx"]]) | {"_score": round(float(r["_score"]), 3)} for r in ranked[:limit]]
    top = ranked[0]["_score"] if ranked else 0.0
    confidence = max(0.0, min(1.0, float(top) / (1.0 + PHRASE_BONUS)))
    tier = "direct" if confidence >= DIRECT_T else "likely" if confidence >= LIKELY_T else "guidance"

    log_event({"endpoint": "/ask", "query": q, "intent": intent, "decision": {"route": "generic-search", "tier": tier, "confidence": confidence}})
    if not out:
        return jsonify({"message": "No matches. Can you confirm size (mm²), cores, and voltage?",
                        "tier": "guidance", "confidence": confidence, "results": []})
    return jsonify({"tier": tier, "confidence": confidence, "results": out})

@app.route("/calc", methods=["GET"])
def calc():
    # Placeholder: voltage drop only, resistive, no reactance/derating.
    if request.args.get("type") != "voltage_drop":
        return jsonify({"error": "unknown calc type"}), 400

    def _req(name):
        v = request.args.get(name)
        if v is None:
            raise ValueError(f"missing required param: {name}")
        try:
            return float(v)
        except ValueError:
            raise ValueError(f"invalid numeric param: {name}={v}")

    try:
        size = _req("size_mm2")
        current = _req("current_a")
        length = _req("length_m")
        material = request.args.get("material", "copper").strip().lower()
        phase = request.args.get("phase", "single").strip().lower()  # single|three|dc
        supply_v = request.args.get("supply_v")  # optional

        if material not in ("copper", "aluminium", "aluminum"):
            return jsonify({"error": f"invalid material: {material}"}), 400

        rho = 0.017241 if material == "copper" else 0.028264  # Ω·mm²/m @20°C
        r_per_m = rho / size  # Ω/m conductor

        if phase in ("single", "dc"):
            path_factor = 2.0           # out-and-back
        elif phase == "three":
            path_factor = 1.7320508075688772  # √3
        else:
            return jsonify({"error": f"invalid phase: {phase}"}), 400

        vd_v = current * r_per_m * length * path_factor
        result = {"voltage_drop_v": vd_v}
        if supply_v is not None:
            try:
                sv = float(supply_v)
                if sv > 0:
                    result["voltage_drop_pct"] = (vd_v / sv) * 100.0
            except ValueError:
                pass

        resp = {
            "result": result,
            "assumptions": {
                "temp_c": 20,
                "rho_ohm_mm2_per_m": rho,
                "phase": phase,
                "material": material,
                "note": "placeholder calc; no reactance, no derating; AS/NZS tables not applied"
            }
        }
        log_event({"endpoint": "/calc", "query": request.query_string.decode(), "intent": "calc", "decision": {"type": "voltage_drop"}})
        return jsonify(resp)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
