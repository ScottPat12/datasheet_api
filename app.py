# app.py
import os
import re
import json
import time
import uuid
from difflib import SequenceMatcher

from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd

# ----------------------------
# Config / Environment
# ----------------------------
ITEMS_CSV_PATH = os.getenv("ITEMS_CSV_PATH", "data/Items.csv")
DIRECT_T = float(os.getenv("DIRECT_T", 0.80))
LIKELY_T = float(os.getenv("LIKELY_T", 0.55))
API_KEY = os.getenv("API_KEY", "")  # optional; if empty, auth disabled
APP_VERSION = os.getenv("APP_VERSION", "v1")

# ----------------------------
# App init
# ----------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # adjust allowlist if needed

_last_data_load_at = None

# ----------------------------
# Data loading
# ----------------------------
def normalize_code(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    # collapse spaces, dashes, underscores, slashes; keep alnum and dots
    s = re.sub(r"[\s\-_/]+", "", s)
    return s

def load_items(csv_path: str) -> pd.DataFrame:
    global _last_data_load_at
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    # normalize Stock Code for robust by_code
    if "Stock Code" not in df.columns:
        raise ValueError("Items.csv must include a 'Stock Code' column")
    df["__norm_code"] = df["Stock Code"].map(normalize_code)

    # build search blob: lower-cased concat of all visible columns
    def row_blob(row):
        vals = []
        for c in df.columns:
            if c.startswith("__"):  # internal columns
                continue
            vals.append(str(row[c]))
        return " ".join(vals).lower()

    df["__search_blob"] = df.apply(row_blob, axis=1)
    _last_data_load_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return df

try:
    _df = load_items(ITEMS_CSV_PATH)
except Exception as e:
    # Defer raising until healthz or first call; app still starts
    _df = pd.DataFrame()
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
# Parsing helpers (numeric)
# ----------------------------
def parse_mm2(s):
    s = (s or "").replace(",", "").lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mm2|mm²|sqmm)?", s)
    return float(m.group(1)) if m else None

def parse_volts(s):
    s = (s or "").replace(" ", "").replace(",", "").lower()
    # handles 1kv, 1.1kv, 1100v, 1100vac, 1100vdc
    m = re.search(r"(\d+(?:\.\d+)?)(kv|v)", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    return val * 1000.0 if unit == "kv" else val

def apply_numeric_filters(row: dict, ops: dict):
    """
    Returns (ok, reasons). Reasons contains short strings for why the row failed.
    Missing numeric value => fail when the relevant comparator is present.
    """
    reasons = []
    size = parse_mm2(row.get("Size") or row.get("Core Size") or "")
    volt = parse_volts(row.get("Voltage") or row.get("Max Voltage") or "")

    def check(val, op, target, label):
        tgt = float(target)
        if val is None:
            reasons.append(f"{label}:missing")
            return False
        if op.endswith("__gte"):
            if val < tgt:
                reasons.append(f"{label}<{tgt}")
                return False
        elif op.endswith("__lte"):
            if val > tgt:
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
# Search / scoring helpers
# ----------------------------
def tokenize(q: str):
    q = (q or "").lower()
    # split on non-alnum and keep mm2-like tokens intact
    return [t for t in re.split(r"[^\w\.]+", q) if t]

def token_score(blob: str, term: str) -> float:
    blob = (blob or "").lower()
    term = (term or "").lower()
    if term in blob:
        return 1.0
    return max(0.0, SequenceMatcher(None, term, blob).ratio() - 0.15)

def relevance(blob: str, terms: list[str]) -> float:
    terms = [t for t in (terms or []) if t]
    if not terms:
        return 0.0
    return sum(token_score(blob, t) for t in terms) / len(terms)

def classify_user_intent(q: str) -> str:
    ql = (q or "").lower()
    if re.search(r"[a-z]\d|/", ql):
        return "product_lookup"
    if any(k in ql for k in ["outdoor", "halogen", "armour", "armored", "armoured", "mm2", "mm²", "kv", "v "]):
        return "attribute_query"
    if any(k in ql for k in ["regulation", "standard", "as/nzs", "code of practice"]):
        return "regulation"
    if any(k in ql for k in ["substitute", "alternative", "equivalent"]):
        return "substitute_search"
    return "general"

# ----------------------------
# Response helpers
# ----------------------------
def df_row_to_item(row) -> dict:
    d = {c: row[c] for c in _df.columns if not c.startswith("__")}
    return d

def suggest_near_codes(norm_q: str, k=3):
    if _df.empty:
        return []
    codes = _df["__norm_code"].tolist()
    sims = [(SequenceMatcher(None, norm_q, c).ratio(), i) for i, c in enumerate(codes)]
    sims.sort(reverse=True)
    out = []
    for score, idx in sims[:k]:
        item = df_row_to_item(_df.iloc[idx])
        out.append({"score": round(float(score), 3), "item": item})
    return out

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
        "last_data_load_at": _last_data_load_at
    })

@app.route("/items/by_code", methods=["GET"])
def by_code():
    if not require_key():
        return jsonify({"error": "unauthorized"}), 401

    q = request.args.get("q", "")
    intent = "product_lookup"
    if not q:
        return jsonify({"error": "missing q"}), 400

    n = normalize_code(q)
    res = _df[_df["__norm_code"] == n]
    if len(res) == 1:
        item = df_row_to_item(res.iloc[0])
        log_event({"endpoint": "/items/by_code", "query": q, "intent": intent, "decision": {"hit": True}})
        return jsonify(item)

    # 404 with suggestions
    suggestions = suggest_near_codes(n, k=3)
    log_event({"endpoint": "/items/by_code", "query": q, "intent": intent, "decision": {"hit": False, "suggestions": len(suggestions)}})
    return jsonify({
        "error": "not_found",
        "message": "No exact code match",
        "suggestions": suggestions
    }), 404

@app.route("/items/search", methods=["GET"])
def search_items():
    if not require_key():
        return jsonify({"error": "unauthorized"}), 401

    q = request.args.get("q", "")
    limit = max(1, min(int(request.args.get("limit", "5")), 10))
    terms = tokenize(q)
    intent = classify_user_intent(q)

    results = []
    if not _df.empty and terms:
        for idx, row in _df.iterrows():
            sc = relevance(row["__search_blob"], terms)
            if sc > 0:
                results.append({"_score": sc, "_idx": idx})

    results.sort(key=lambda r: r["_score"], reverse=True)
    top = results[0]["_score"] if results else 0.0
    confidence = max(0.0, min(1.0, float(top)))
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

    # Accept query params (GET) or JSON (POST) with the same shape
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
        # allow CSV
        if isinstance(inc, list) and len(inc) == 1:
            inc = inc[0]
        if isinstance(exc, list) and len(exc) == 1:
            exc = exc[0]
        include = [t.strip().lower() for t in (inc.split(",") if isinstance(inc, str) else inc) if t.strip()]
        exclude = [t.strip().lower() for t in (exc.split(",") if isinstance(exc, str) else exc) if t.strip()]
        limit = int(request.args.get("limit", 10))
        # numeric operators from query params
        for k in ("size__gte", "size__lte", "voltage__gte", "voltage__lte"):
            if k in request.args:
                numeric_ops[k] = request.args.get(k)

    limit = max(1, min(limit, 25))

    def row_passes(row):
        blob = row["__search_blob"]
        # include tokens must appear
        for t in include:
            if t not in blob:
                return False, [f"missing:{t}"]
        # exclude tokens must not appear
        for t in exclude:
            if t in blob:
                return False, [f"excluded:{t}"]
        ok, reasons = apply_numeric_filters({c: row[c] for c in _df.columns if not c.startswith("__")}, numeric_ops)
        return ok, reasons

    candidates = len(_df.index)
    passed = []
    why_out_collected = []

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

    return jsonify({
        "results": out,
        "filter_summary": summary
    })

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

    # simple router
    if intent == "product_lookup":
        # Try exact by_code first
        n = normalize_code(q)
        res = _df[_df["__norm_code"] == n]
        if len(res) == 1:
            item = df_row_to_item(res.iloc[0])
            log_event({"endpoint": "/ask", "query": q, "intent": intent, "decision": {"route": "by_code", "hit": True}})
            return jsonify({"tier": "direct", "confidence": 0.95, "results": [item]})
        # fallback to search
        terms = tokenize(q)
        ranked = []
        for idx, row in _df.iterrows():
            sc = relevance(row["__search_blob"], terms)
            if sc > 0:
                ranked.append({"_score": sc, "_idx": idx})
        ranked.sort(key=lambda r: r["_score"], reverse=True)
        out = [df_row_to_item(_df.iloc[r["_idx"]]) | {"_score": round(float(r["_score"]), 3)} for r in ranked[:limit]]
        conf = round(float(ranked[0]["_score"]), 3) if ranked else 0.0
        tier = "direct" if conf >= DIRECT_T else "likely" if conf >= LIKELY_T else "guidance"
        log_event({"endpoint": "/ask", "query": q, "intent": intent, "decision": {"route": "search", "tier": tier, "confidence": conf}})
        if not out:
            return jsonify({"message": "I couldn't find an exact match. Do you have a specific code or size/cores?",
                            "tier": "guidance", "confidence": conf, "results": []})
        return jsonify({"tier": tier, "confidence": conf, "results": out})

    if intent in ("attribute_query", "substitute_search"):
        # Map to filter-like: treat tokens as include
        include = [t for t in tokenize(q) if t]
        request_args = request.args.to_dict()
        # Allow numeric hints in text (e.g., 4mm2, 1kv)
        numeric_ops = {}
        # naive parse: if "4mm2" in text, interpret as min size
        for t in include:
            m1 = re.match(r"(\d+(?:\.\d+)?)mm2|mm²|sqmm", t)
            if m1:
                numeric_ops["size__gte"] = m1.group(1)
            m2 = re.match(r"(\d+(?:\.\d+)?)(kv|v)", t)
            if m2:
                v = float(m2.group(1)) * (1000 if m2.group(2) == "kv" else 1)
                numeric_ops["voltage__gte"] = str(v)

        # evaluate filter
        def row_passes(row):
            blob = row["__search_blob"]
            for t in include:
                if t not in blob:
                    return False
            ok, _ = apply_numeric_filters({c: row[c] for c in _df.columns if not c.startswith("__")}, numeric_ops)
            return ok

        hits = []
        for idx, row in _df.iterrows():
            if row_passes(row):
                hits.append(idx)
        out = [df_row_to_item(_df.iloc[i]) for i in hits[:limit]]
        tier = "direct" if len(out) else "guidance"
        conf = 0.85 if len(out) else 0.3
        log_event({"endpoint": "/ask", "query": q, "intent": intent, "decision": {"route": "filter-like", "tier": tier, "confidence": conf}})
        if not out:
            return jsonify({"message": "No exact matches. Can you confirm size (mm²), cores, and voltage?",
                            "tier": "guidance", "confidence": conf, "results": []})
        return jsonify({"tier": tier, "confidence": conf, "results": out})

    # generic fallback: search
    terms = tokenize(q)
    ranked = []
    for idx, row in _df.iterrows():
        sc = relevance(row["__search_blob"], terms)
        if sc > 0:
            ranked.append({"_score": sc, "_idx": idx})
    ranked.sort(key=lambda r: r["_score"], reverse=True)
    out = [df_row_to_item(_df.iloc[r["_idx"]]) | {"_score": round(float(r["_score"]), 3)} for r in ranked[:limit]]
    conf = round(float(ranked[0]["_score"]), 3) if ranked else 0.0
    tier = "direct" if conf >= DIRECT_T else "likely" if conf >= LIKELY_T else "guidance"
    log_event({"endpoint": "/ask", "query": q, "intent": intent, "decision": {"route": "search", "tier": tier, "confidence": conf}})
    if not out:
        return jsonify({"message": "I couldn’t determine exactly what you need. Do you have a code or key attributes (size, cores, voltage)?",
                        "tier": "guidance", "confidence": conf, "results": []})
    return jsonify({"tier": tier, "confidence": conf, "results": out})

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
