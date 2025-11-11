from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

# Load items from CSV
DATA_PATH = os.getenv("ITEMS_CSV_PATH", "data/Items.csv")

try:
    df_items = pd.read_csv(DATA_PATH, dtype=str).fillna("")
except Exception as e:
    raise RuntimeError(f"Failed to load items file from {DATA_PATH}: {e}")

# Fields to search through
SEARCHABLE_FIELDS = [
    "Stock Code",
    "Name",
    "Short description",
    "Marketing description",
    "Applications",
    "Benefits"
]

@app.route("/search_items")
def search_items():
    """Search for items by query string across specific fields"""
    query = request.args.get("q", "").strip().lower()
    limit = int(request.args.get("limit", 5))

    if not query:
        return jsonify({"error": "Missing 'q' search parameter"}), 400

    results = []
    for _, row in df_items.iterrows():
        for field in SEARCHABLE_FIELDS:
            if query in row.get(field, "").lower():
                results.append(row.to_dict())
                break
        if len(results) >= limit:
            break

    return jsonify({
        "query": query,
        "count": len(results),
        "results": results
    })

@app.route("/healthz")
def health_check():
    """Basic health check endpoint"""
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
