# app.py
import os, re, json, time, uuid, math
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional

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

# Scoring knobs (generic)
PHRASE_MAX_N = 4
PHRASE_BONUS = 1.0
PROX_WINDOW = 3
PROX_BONUS = 0.25
FUZZY_TERM_PENALTY = 0.15
FIELD_WEIGHTS = {
    "Stock Code": 2.5,
    "Stock Group": 2.0,
    "Name": 1.8,
    "Category": 1.4,
    "Series name": 2.2,
    "Short description": 1.6,
    "Marketing description": 1.4,
}

# Session tips (optional, very light)
_SESSION_LOWCONF = {}

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
    if not s: return ""
    s = s.strip().lower()
    return re.sub(r"[\s\-_/]+", "", s)

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
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mm2|mm²|sqmm)", s)
    return float(m.group(1)) if m else None

def parse_volts(s):
    s = (s or "").replace(" ", "").replace(",", "").lower()
    m = re.search(r"(\d+(?:\.\d+)?)(kv|v)", s)
    if not m: return None
    val = float(m.group(1)); unit = m.group(2)
    return val * 1000.0 if unit == "kv" else val

def session_id() -> str:
    return request.headers.get("X-Session-ID") or request.remote_addr or "anon"

# ----------------------------
# Optional Series table
# ----------------------------
def try_load_series(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype=str).fillna("")
    # normalize headers
    colmap = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("stock group", "stock_group"): colmap[c] = "Stock Group"
        elif cl in ("series name", "series", "series_name"): colmap[c] = "Series name"
        elif cl in ("short description", "short_desc", "short"): colmap[c] = "Short description"
        elif cl in ("marketing description", "marketing_desc", "marketing"): colmap[c] = "Marketing description"
    if colmap: df = df.rename(columns=colmap)
    keep = [c for c in ["Stock Group","Series name","Short description","Marketing description"] if c in df.columns]
    if not keep or "Stock Group" not in keep:
        return None
    return df[keep].drop_duplicates(subset=["Stock Group"]).copy()

# ----------------------------
# Data loading + corpus build
# ----------------------------
def load_items(items_csv: str, series_csv: Optional[str]) -> Tuple[pd.DataFrame, dict]:
    global _last_data_load_at
    items = pd.read_csv(items_csv, dtype=str).fillna("")
    if "Stock Code" not in items.columns:
        raise ValueError("Items.csv must include a 'Stock Code' column")
    if "Stock Group" not in items.columns:
        items["Stock Group"] = ""

    s_df = try_load_series(series_csv) if series_csv else None
    if s_df is not None:
        items = items.merge(s_df, on="Stock Group", how="left")
    else:
        for c in ["Series name","Short description","Marketing description"]:
            if c not in items.columns: items[c] = ""

    items["__norm_code"] = items["Stock Code"].map(normalize_code)
    items["__code_prefix"] = items["Stock Code"].map(code_prefix)

    # lowercase copies
    field_texts = {}
    for col in items.columns:
        if col.startswith("__"): continue
        field_texts[col] = items[col].astype(str).str.lower().fillna("")

    unified, tokens, per_field_tokens = [], [], []
    for i in range(len(items)):
        parts = []
        for col in items.columns:
            if col.startswith("__"): continue
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

    # IDF
    items["_tmp_all_tokens"] = items["__tokens"].apply(set)
    docfreq = Counter()
    for s in items["_tmp_all_tokens"]:
        docfreq.update(s)
    N = len(items)
    idf = {term: math.log((1 + N) / (1 + dfreq)) + 1.0 for term, dfreq in docfreq.items()}
    items.drop(columns=["_tmp_all_tokens"], inplace=True)

    _last_data_load_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return items, idf

try:
    _df, _idf = load_items(ITEMS_CSV_PATH, SERIES_CSV_PATH)
except Exception as e:
    _df = pd.DataFrame(); _idf = {}; _data_load_error = str(e)
else:
    _data_load_error = ""

# ----------------------------
# Logging
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
# Security
# ----------------------------
def require_key():
    if not API_KEY: return True
    from hmac import compare_digest
    return compare_digest(request.headers.get("X-API-Key", ""), API_KEY)

# ----------------------------
# Numeric filters
# ----------------------------
def apply_numeric_filters(row: dict, ops: dict) -> Tuple[bool, List[str]]:
    reasons = []
    size = parse_mm2(row.get("Size") or row.get("Core Size") or "")
    volt = parse_volts(row.get("Voltage") or row.get("Max Voltage") or "")
    def check(val, op, target, label):
        tgt = float(target)
        if val is None:
            reasons.append(f"{label}:missing"); return False
        if op.endswith("__gte") and val < tgt:
            reasons.append(f"{label}<{tgt}"); return False
        if op.endswith("__lte") and val > tgt:
            reasons.append(f"{label}>{tgt}"); return False
        return True
    ok = True
    if "size__gte" in ops: ok &= check(size, "size__gte", ops["size__gte"], "size")
    if "size__lte" in ops: ok &= check(size, "size__lte", ops["size__lte"], "size")
    if "voltage__gte" in ops: ok &= check(volt, "voltage__gte", ops["voltage__gte"], "voltage")
    if "voltage__lte" in ops: ok &= check(volt, "voltage__lte", ops["voltage__lte"], "voltage")
    return ok, reasons

# ----------------------------
# Scoring (generic)
# ----------------------------
def tfidf_score(row_tokens, query_tokens):
    tf = Counter(row_tokens)
    num = den_row = den_q = 0.0
    q_vec = {}
    for t in query_tokens:
        w = _idf.get(t, 1.0)
        q_vec[t] = q_vec.get(t, 0.0) + w
    for t, q_w in q_vec.items():
        d_w = (tf.get(t, 0) * _idf.get(t, 1.0))
        num += q_w * d_w
        den_row += d_w * d_w
        den_q += q_w * q_w
    if den_row == 0.0 or den_q == 0.0: return 0.0
    return num / math.sqrt(den_row * den_q)

def phrase_bonus(row_blob: str, q_tokens):
    if len(q_tokens) < 2: return 0.0
    bonus = 0.0
    for n in range(2, min(PHRASE_MAX_N, len(q_tokens)) + 1):
        for p in ngrams(q_tokens, n):
            if p in row_blob: bonus += PHRASE_BONUS
    return bonus

def proximity_bonus(row_tokens, q_tokens):
    uniq = list(dict.fromkeys(q_tokens))
    pos = defaultdict(list)
    for i, tok in enumerate(row_tokens): pos[tok].append(i)
    bonus = 0.0
    for i in range(len(uniq)):
        for j in range(i+1, len(uniq)):
            a, b = uniq[i], uniq[j]
            if not pos[a] or not pos[b]: continue
            md = min(abs(pa - pb) for pa in pos[a] for pb in pos[b])
            if md <= PROX_WINDOW: bonus += PROX_BONUS
    return bonus

def field_weight_bonus(field_tokens_map, q_tokens):
    bonus = 0.0; qset = set(q_tokens)
    for field, toks in field_tokens_map.items():
        if not toks: continue
        w = FIELD_WEIGHTS.get(field)
        if w and qset.intersection(toks): bonus += w * 0.15
    return bonus

def fuzzy_term_adjust(row_blob: str, term: str) -> float:
    if term in row_blob: return 0.0
    s = SequenceMatcher(None, term, row_blob).ratio()
    return max(0.0, s - FUZZY_TERM_PENALTY)

def tokenize_query(q: str) -> List[str]:
    q = (q or "").strip().lower()
    toks = [t for t in TOKEN_SPLIT_RE.split(q) if t]
    return [t for t in toks if len(t) > 1]

def generic_score(row, q_tokens):
    base = tfidf_score(row["__tokens"], q_tokens)
    p = phrase_bonus(row["__search_blob"], q_tokens)
    prox = proximity_bonus(row["__tokens"], q_tokens)
    f = field_weight_bonus(row["__field_tokens"], q_tokens)
    fz = sum(fuzzy_term_adjust(row["__search_blob"], t) for t in q_tokens)
    return base + p + prox + f + 0.1 * fz

# ----------------------------
# Converters / responses
# ----------------------------
def df_row_to_item(row) -> dict:
    return {c: row[c] for c in _df.columns if not c.startswith("__")}

def suggest_near_codes(norm_q: str, k=3):
    if _df.empty: return []
    sims = [(SequenceMatcher(None, norm_q, c).ratio(), i) for i, c in enumerate(_df["__norm_code"].tolist())]
    sims.sort(reverse=True)
    out = []
    for score, idx in sims[:k]:
        out.append({"score": round(float(score), 3), "item": df_row_to_item(_df.iloc[idx])})
    return out

def run_generic_search(query: str, limit: int = 5):
    q_tokens = tokenize_query(query)
    ranked = []
    for idx, row in _df.iterrows():
        sc = generic_score(row, q_tokens)
        if sc > 0:
            ranked.append({"_score": sc, "_idx": idx})
    ranked.sort(key=lambda r: r["_score"], reverse=True)
    top = ranked[0]["_score"] if ranked else 0.0
    confidence = max(0.0, min(1.0, float(top) / (1.0 + PHRASE_BONUS)))
    tier = "direct" if confidence >= DIRECT_T else "likely" if confidence >= LIKELY_T else "guidance"
    out = [df_row_to_item(_df.iloc[r["_idx"]]) | {"_score": round(float(r["_score"]), 3)} for r in ranked[:limit]]
    return {"tier": tier, "confidence": round(confidence, 3), "results": out}

def run_filter_on_df(include: List[str], exclude: List[str], numeric_ops: Dict[str, str], limit: int = 10):
    include = [t.strip().lower() for t in include if t.strip()]
    exclude = [t.strip().lower() for t in exclude if t.strip()]
    passed, why_out = [], []
    for idx, row in _df.iterrows():
        blob = row["__search_blob"]
        ok = True; reasons = []
        for t in include:
            if t not in blob: ok = False; reasons.append(f"missing:{t}")
        for t in exclude:
            if t in blob: ok = False; reasons.append(f"excluded:{t}")
        if ok:
            ok2, r2 = apply_numeric_filters(df_row_to_item(row), numeric_ops)
            ok = ok and ok2; reasons.extend(r2 if not ok2 else [])
        if ok: passed.append(idx)
        elif len(why_out) < 10 and reasons: why_out.append(";".join(reasons))
    out_rows = passed[:max(1, min(limit, 25))]
    out = [df_row_to_item(_df.iloc[i]) for i in out_rows]
    return {
        "results": out,
        "filter_summary": {
            "include": include, "exclude": exclude, "numeric": numeric_ops,
            "candidates": len(_df.index), "returned": len(out),
            "why_out_sample": why_out[:5]
        }
    }

# ----------------------------
# Intent front-door (classifier + entities)
# ----------------------------
INTENT_KEYWORDS = {
    "compare":       [r"\bcompare\b", r"\bdifference\b", r"\bvs\b", r"\bversus\b"],
    "product_lookup":[r"\b(stock|code|part)\b", r"\bwhat is\b"],
    "attribute_lookup":[r"\bvoltage\b", r"\brating\b", r"\buv\b", r"\btemperature\b", r"\bdimension\b", r"\bod\b", r"\bcores?\b"],
    "criteria_list": [r"\bshow\b", r"\blist\b", r"\bfind\b", r"\bat least\b", r"\bunder\b", r"\bless than\b", r"\bgreater than\b", r"≥|<=|between"],
    "procedural":    [r"\bhow to\b", r"\binstall\b", r"\bterminate\b"],
    "relational":    [r"\bplug\b", r"\bconnector\b", r"\bfits\b", r"\bcompatible\b"],
}
SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mm2|mm²|sqmm)", re.I)
VOLT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(kv|v|volt)", re.I)
CORES_RE = re.compile(r"(\d+)\s*(core|cores|g)", re.I)
CODE_RE  = re.compile(r"[A-Z]{1,}[0-9A-Z\-\/\.]*[0-9](?:G[0-9\.]+)?", re.I)
MATERIAL_TERMS = ["rubber", "epr", "xlpe", "pvc", "lszh", "halogen free", "halogen-free"]
ENV_TERMS      = ["outdoor", "indoor", "underground", "buried", "uv", "sunlight", "wet", "oil", "chemical", "neutral screen", "screen"]

def extract_entities(q: str) -> Dict[str, Any]:
    ents: Dict[str, Any] = {"codes": [], "attributes": {}, "terms": []}
    if m := SIZE_RE.search(q):   ents["attributes"]["size_mm2"] = float(m.group(1))
    if m := VOLT_RE.search(q):
        v = float(m.group(1)) * (1000 if m.group(2).lower().startswith("kv") else 1)
        ents["attributes"]["voltage_v"] = v
    if m := CORES_RE.search(q):  ents["attributes"]["cores"] = int(m.group(1))
    codes = CODE_RE.findall(q)
    ents["codes"] = [c if isinstance(c, str) else c[0] for c in codes]
    ql = q.lower()
    ents["terms"] += [t for t in MATERIAL_TERMS if t in ql]
    ents["terms"] += [t for t in ENV_TERMS if t in ql]
    toks = [t for t in tokenize_query(q) if not re.fullmatch(r"\d+(?:\.\d+)?", t)]
    ents["terms"] += list(dict.fromkeys(toks))
    return ents

def classify_and_parse(q: str) -> Dict[str, Any]:
    ql = q.lower()
    scores = {k: 0 for k in INTENT_KEYWORDS}
    for intent, patterns in INTENT_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, ql): scores[intent] += 1
    if CODE_RE.search(q): scores["product_lookup"] += 2
    intent = max(scores, key=scores.get)
    confidence = 0.2 + 0.2 * scores[intent]
    ents = extract_entities(q)
    needs = (intent in {"product_lookup","compare"} and not ents["codes"])
    return {
        "intent": intent if scores[intent] > 0 else "general",
        "entities": ents,
        "confidence": round(min(confidence, 0.95), 2),
        "needs_clarification": bool(needs)
    }

# Attribute extraction
ATTR_HINTS = {
    "voltage": ["Voltage", "Max Voltage", "Rated Voltage"],
    "temperature": ["Temperature", "Max Temp", "Operating Temperature"],
    "uv": ["UV", "UV stabilised", "UV stabilized"],
    "od": ["OD", "Outer Diameter", "Overall Diameter"],
    "cores": ["Cores", "Core", "No. Cores"],
    "size": ["Size", "Core Size", "Conductor Size", "Cross Section"],
    "screen": ["Screen", "Shield", "Neutral Screen", "EMC"],
}

def extract_attributes_for_item(item: Dict[str, Any], query: str) -> Dict[str, Any]:
    ql = query.lower()
    picked = {}
    for key, cols in ATTR_HINTS.items():
        if key in ql or any(word in ql for word in key.split()):
            for c in cols:
                if c in item and str(item[c]).strip():
                    picked[c] = item[c]
    for c in ["Stock Code", "Name", "Stock Group", "Category", "Data sheet"]:
        if c in item and str(item[c]).strip():
            picked.setdefault(c, item[c])
    return picked or item

def compare_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items: return {"diff": {}, "items": []}
    common_cols = set(items[0].keys())
    for it in items[1:]:
        common_cols &= set(it.keys())
    ignore = {c for c in common_cols if c.startswith("__")}
    diff = {}
    for c in sorted(common_cols - ignore):
        vals = [str(it.get(c, "")).strip() for it in items]
        if len(set(vals)) > 1:
            diff[c] = vals
    return {"diff": diff, "items": items}

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

@app.route("/help", methods=["GET"])
def help_examples():
    examples = [
        "What is MC05/2.5?",
        "MC05/2.5 voltage rating",
        "Find outdoor halogen-free 4 mm² cables",
        "Compare MC05/2.5 vs MCN05/2.5",
        "Voltage drop for 6 mm², 20 A, 20 m copper"
    ]
    msg = (
        "Ask about product codes, specs, or cable types.\n\n"
        "Examples:\n- " + "\n- ".join(examples)
    )
    return jsonify({"message": msg, "examples": examples})

@app.route("/items/by_code", methods=["GET"])
def by_code():
    if not require_key(): return jsonify({"error": "unauthorized"}), 401
    q = request.args.get("q", "")
    if not q: return jsonify({"error": "missing q"}), 400
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
    if not require_key(): return jsonify({"error": "unauthorized"}), 401
    q = request.args.get("q", "")
    limit = max(1, min(int(request.args.get("limit", "5")), 10))
    res = run_generic_search(q, limit=limit)
    log_event({"endpoint": "/items/search", "query": q, "intent": "attribute_query",
               "decision": {"tier": res["tier"], "confidence": res["confidence"], "returned": len(res["results"])}})
    return jsonify({"query": q, "intent": "attribute_query", **res})

@app.route("/items/filter", methods=["GET","POST"])
def filter_items():
    if not require_key(): return jsonify({"error":"unauthorized"}), 401
    include = []; exclude = []; numeric_ops = {}; limit = 10
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
        for k in ("size__gte","size__lte","voltage__gte","voltage__lte"):
            if k in request.args: numeric_ops[k] = request.args.get(k)
    out = run_filter_on_df(include, exclude, numeric_ops, limit=limit)
    tier = "direct" if out["results"] else "guidance"
    conf = 0.8 if out["results"] else 0.3
    log_event({"endpoint": "/items/filter", "query": "N/A", "intent": "attribute_query",
               "filters": out["filter_summary"], "decision": {"tier": tier, "confidence": conf}})
    return jsonify(out)

@app.route("/ask", methods=["POST"])
def ask():
    if not require_key(): return jsonify({"error": "unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    q = (body.get("query") or "").strip()
    limit = max(1, min(int(body.get("limit", 5)), 10))
    if not q: return jsonify({"error": "missing query"}), 400
    res = run_generic_search(q, limit=limit)
    log_event({"endpoint": "/ask", "query": q, "intent": "general",
               "decision": {"tier": res["tier"], "confidence": res["confidence"]}})
    return jsonify(res)

# ----------------------------
# NEW: Intent Router
# ----------------------------
@app.route("/router/ask", methods=["POST"])
def router_ask():
    if not require_key(): return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    limit = max(1, min(int(data.get("limit", 5)), 10))
    if not query: return jsonify({"error": "Missing query"}), 400

    intent_struct = classify_and_parse(query)
    intent = intent_struct["intent"]
    ents = intent_struct["entities"]

    # product_lookup: try codes in order, else fallback to search
    if intent == "product_lookup":
        for raw in ents["codes"]:
            norm = normalize_code(raw)
            hit = _df[_df["__norm_code"] == norm]
            if not hit.empty:
                item = df_row_to_item(hit.iloc[0])
                if any(k in query.lower() for k in ATTR_HINTS.keys()):
                    attr = extract_attributes_for_item(item, query)
                    payload = {"intent": intent_struct, "result": attr, "source": "by_code+attributes"}
                else:
                    payload = {"intent": intent_struct, "result": item, "source": "by_code"}
                log_event({"endpoint": "/router/ask", "query": query, "intent": intent, "decision": {"route": "by_code"}})
                return jsonify(payload)
        # no exact code → ranked search
        sr = run_generic_search(query, limit=limit)
        log_event({"endpoint": "/router/ask", "query": query, "intent": intent, "decision": {"route": "search"}})
        if sr["confidence"] < 0.5:
            hint = "Tip: include an exact code (e.g. 'MC05/2.5') or add features (e.g. '4 mm² outdoor'). Type 'help' for examples."
            return jsonify({"intent": intent_struct, "results": sr["results"], "tier": sr["tier"], "confidence": sr["confidence"], "message": hint, "source": "search"})
        return jsonify({"intent": intent_struct, "results": sr["results"], "tier": sr["tier"], "confidence": sr["confidence"], "source": "search"})

    # attribute lookup without explicit code → ranked list or single-item attr extraction
    if intent == "attribute_lookup":
        if len(ents["codes"]) == 1:
            norm = normalize_code(ents["codes"][0])
            hit = _df[_df["__norm_code"] == norm]
            if not hit.empty:
                item = df_row_to_item(hit.iloc[0])
                attr = extract_attributes_for_item(item, query)
                log_event({"endpoint": "/router/ask", "query": query, "intent": intent, "decision": {"route": "by_code+attributes"}})
                return jsonify({"intent": intent_struct, "result": attr, "source": "by_code+attributes"})
        sr = run_generic_search(query, limit=limit)
        log_event({"endpoint": "/router/ask", "query": query, "intent": intent, "decision": {"route": "search"}})
        if sr["confidence"] < 0.5:
            hint = "Tip: add a code (e.g. 'MC05/2.5 voltage rating') or include size/voltage (e.g. '4 mm² 1 kV'). Type 'help' for examples."
            return jsonify({"intent": intent_struct, "results": sr["results"], "tier": sr["tier"], "confidence": sr["confidence"], "message": hint, "source": "search"})
        return jsonify({"intent": intent_struct, "results": sr["results"], "tier": sr["tier"], "confidence": sr["confidence"], "source": "search"})

    # criteria_list → filter
    if intent == "criteria_list":
        include = ents["terms"]
        numeric = {}
        if "size_mm2" in ents["attributes"]: numeric["size__gte"] = ents["attributes"]["size_mm2"]
        if "voltage_v" in ents["attributes"]: numeric["voltage__gte"] = ents["attributes"]["voltage_v"]
        out = run_filter_on_df(include, [], numeric, limit=limit)
        tier = "direct" if out["results"] else "guidance"
        conf = 0.8 if out["results"] else 0.3
        log_event({"endpoint": "/router/ask", "query": query, "intent": intent, "decision": {"route": "filter", "tier": tier, "confidence": conf}})
        if not out["results"]:
            hint = "No exact matches. Add/adjust terms (e.g. 'screened', 'rubber') or numbers (e.g. size/voltage). Type 'help' for examples."
            return jsonify({"intent": intent_struct, "results": [], "filter_summary": out["filter_summary"], "tier": tier, "confidence": conf, "message": hint, "source": "filter"})
        return jsonify({"intent": intent_struct, "results": out["results"], "filter_summary": out["filter_summary"], "tier": tier, "confidence": conf, "source": "filter"})

    # compare → return items and diffs
    if intent == "compare":
        picked = []
        for raw in ents["codes"][:6]:
            norm = normalize_code(raw)
            hit = _df[_df["__norm_code"] == norm]
            if not hit.empty:
                picked.append(df_row_to_item(hit.iloc[0]))
        cmpres = compare_items(picked)
        log_event({"endpoint": "/router/ask", "query": query, "intent": intent, "decision": {"route": "compare", "items": len(picked)}})
        if len(picked) < 2:
            hint = "To compare, include two codes — e.g. 'Compare MC05/2.5 vs MCN05/2.5'."
            return jsonify({"intent": intent_struct, "comparison": cmpres, "message": hint, "source": "compare"})
        return jsonify({"intent": intent_struct, "comparison": cmpres, "source": "compare"})

    # procedural/relational → stub (future RAG)
    if intent in {"procedural", "relational"}:
        log_event({"endpoint": "/router/ask", "query": query, "intent": intent, "decision": {"route": "placeholder"}})
        return jsonify({
            "intent": intent_struct,
            "message": "I can answer product/spec queries now. Installation and compatibility answers will come from manuals soon. Try a code or key attributes (size, voltage, environment).",
            "source": "placeholder"
        })

    # default fallback → search + auto-tip on low confidence
    sr = run_generic_search(query, limit=limit)
    log_event({"endpoint": "/router/ask", "query": query, "intent": "general", "decision": {"route": "search"}})

    sid = session_id()
    if sr["confidence"] < 0.5:
        _SESSION_LOWCONF[sid] = _SESSION_LOWCONF.get(sid, 0) + 1
        base_tip = "Tip: Ask for a code (e.g. 'MC05/2.5') or describe features (e.g. '4 mm² outdoor cable'). Type 'help' for examples."
        if _SESSION_LOWCONF[sid] >= 2:
            base_tip = "Need help? Try 'help' or ask: 'Compare MC05/2.5 vs MCN05/2.5' / 'Find 4 mm² halogen-free outdoor cables'."
        return jsonify({"intent": intent_struct, "results": sr["results"], "tier": sr["tier"], "confidence": sr["confidence"], "message": base_tip, "source": "search"})
    return jsonify({"intent": intent_struct, "results": sr["results"], "tier": sr["tier"], "confidence": sr["confidence"], "source": "search"})

# ----------------------------
# Calc (placeholder voltage drop)
# ----------------------------
@app.route("/calc", methods=["GET"])
def calc():
    if request.args.get("type") != "voltage_drop":
        return jsonify({"error": "unknown calc type"}), 400
    def _req(name):
        v = request.args.get(name)
        if v is None: raise ValueError(f"missing required param: {name}")
        try: return float(v)
        except ValueError: raise ValueError(f"invalid numeric param: {name}={v}")
    try:
        size = _req("size_mm2"); current = _req("current_a"); length = _req("length_m")
        material = request.args.get("material", "copper").strip().lower()
        phase = request.args.get("phase", "single").strip().lower()  # single|three|dc
        supply_v = request.args.get("supply_v")
        if material not in ("copper","aluminium","aluminum"):
            return jsonify({"error": f"invalid material: {material}"}), 400
        rho = 0.017241 if material == "copper" else 0.028264  # Ω·mm²/m @20°C
        r_per_m = rho / size
        if phase in ("single","dc"): path_factor = 2.0
        elif phase == "three": path_factor = 1.7320508075688772
        else: return jsonify({"error": f"invalid phase: {phase}"}), 400
        vd_v = current * r_per_m * length * path_factor
        result = {"voltage_drop_v": vd_v}
        if supply_v:
            try:
                sv = float(supply_v)
                if sv > 0: result["voltage_drop_pct"] = (vd_v / sv) * 100.0
            except ValueError:
                pass
        log_event({"endpoint": "/calc", "query": request.query_string.decode(), "intent": "calc", "decision": {"type": "voltage_drop"}})
        return jsonify({"result": result, "assumptions": {"temp_c":20,"rho_ohm_mm2_per_m":rho,"phase":phase,"material":material,"note":"placeholder calc; no reactance/derating; AS/NZS tables not applied"}})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
