import os, json
import pandas as pd

# ----------- CONFIG: file names & destinations ------------
CABLES_XLSX = "Cables.xlsx"
INSTALL_XLSX = "Installation Conditions.xlsx"
RANK_XLSX = "Ranking.xlsx"

OUT_DIR = "data"
OUT_CABLES_JSON = os.path.join(OUT_DIR, "cables.json")
OUT_INSTALL_JSON = os.path.join(OUT_DIR, "install_conditions.json")
OUT_RANK_JSON = os.path.join(OUT_DIR, "ranking.json")

os.makedirs(OUT_DIR, exist_ok=True)

# ----------- Helpers --------------------------------------
def to_float(x):
    try:
        if pd.isna(x): return None
        s = str(x).strip().replace(",", "")
        return float(s)
    except:
        return None

def normalize_str(x):
    return "" if pd.isna(x) else str(x).strip()

# ----------- CABLES: read & map columns -------------------
# Expected columns in Cables.xlsx (header row):
#   Series, Code, Cores, Core Size, Cable Type, Armour,
#   CCC (A) [optional], R_ohm_per_km [optional], X_ohm_per_km [optional], Datasheet URL [optional]
#
# NOTE: If CCC/R/X aren’t present yet, leave them blank; rows without these will be skipped by /calc.

print("Reading:", CABLES_XLSX)
df_c = pd.read_excel(CABLES_XLSX, sheet_name=0, dtype=str)
df_c = df_c.fillna("")

# Flexible header matching
def pick(df, *names):
    for n in names:
        if n in df.columns: return n
    return None

col_series = pick(df_c, "Series")
col_code   = pick(df_c, "Code")
col_cores  = pick(df_c, "Cores", "NumCores")
col_size   = pick(df_c, "Core Size", "CoreSize", "Size_mm2")
col_type   = pick(df_c, "Cable Type", "Type")
col_armour = pick(df_c, "Armour")

col_ccc = pick(df_c, "CCC (A)", "CCC", "Capacity_A")
col_r   = pick(df_c, "R_ohm_per_km", "R (ohm/km)", "Resistance_ohm_per_km")
col_x   = pick(df_c, "X_ohm_per_km", "X (ohm/km)", "Reactance_ohm_per_km")
col_url = pick(df_c, "Datasheet URL", "Data sheet", "Datasheet", "URL")

required = [col_series, col_code, col_cores, col_size, col_type, col_armour]
if any(c is None for c in required):
    missing = [n for c,n in zip(required,["Series","Code","Cores","Core Size","Cable Type","Armour"]) if c is None]
    raise RuntimeError(f"Missing required columns in {CABLES_XLSX}: {missing}")

cables = []
for _, r in df_c.iterrows():
    item = {
        "series": normalize_str(r[col_series]),
        "code": normalize_str(r[col_code]),
        "cores": to_float(r[col_cores]),
        "size_mm2": to_float(r[col_size]),
        "type": normalize_str(r[col_type]).lower(),       # normalize for filters
        "armour": normalize_str(r[col_armour]).lower(),   # normalize for filters
        "ccc_a": to_float(r[col_ccc]) if col_ccc else None,
        "r_ohm_per_km": to_float(r[col_r]) if col_r else None,
        "x_ohm_per_km": to_float(r[col_x]) if col_x else None,
        "datasheet_url": normalize_str(r[col_url]) if col_url else ""
    }
    # Keep all rows; /calc can skip those without CCC/R/X
    cables.append(item)

with open(OUT_CABLES_JSON, "w", encoding="utf-8") as f:
    json.dump(cables, f, ensure_ascii=False, indent=2)
print(f"Wrote {OUT_CABLES_JSON} ({len(cables)} rows)")

# ----------- INSTALL CONDITIONS: id -> factors -------------
# Expected columns (minimal):
#   InstallID, temp, group, (any additional numeric factors OK)
print("Reading:", INSTALL_XLSX)
df_i = pd.read_excel(INSTALL_XLSX, sheet_name=0, dtype=str).fillna("")
# Pick first column as ID if we don’t know header names
id_col = None
for guess in ["InstallID", "ID", "Install Cond", "InstallCondID"]:
    if guess in df_i.columns:
        id_col = guess
        break
if id_col is None:
    # fallback: first column
    id_col = df_i.columns[0]

install = {}
for _, r in df_i.iterrows():
    key = normalize_str(r[id_col])
    if not key: continue
    row = {}
    for c in df_i.columns:
        if c == id_col: continue
        val = r[c]
        f = to_float(val)
        if f is not None:
            row[c] = f
    install[key] = row

with open(OUT_INSTALL_JSON, "w", encoding="utf-8") as f:
    json.dump(install, f, ensure_ascii=False, indent=2)
print(f"Wrote {OUT_INSTALL_JSON} ({len(install)} ids)")

# ----------- RANKING: series -> integer rank ---------------
# Expected columns:
#   Series, Rank
print("Reading:", RANK_XLSX)
df_r = pd.read_excel(RANK_XLSX, sheet_name=0, dtype=str).fillna("")
col_s = "Series" if "Series" in df_r.columns else df_r.columns[0]
col_rank = None
for guess in ["Rank", "ranking", "Weight", "Order"]:
    if guess in df_r.columns:
        col_rank = guess
        break
if col_rank is None:
    raise RuntimeError(f"Missing rank column in {RANK_XLSX} (need a column named Rank/Weight/Order)")

ranking = {}
for _, r in df_r.iterrows():
    s = normalize_str(r[col_s])
    rk = r[col_rank]
    if not s: continue
    try:
        ranking[s] = int(float(str(rk).strip()))
    except:
        ranking[s] = 9999

with open(OUT_RANK_JSON, "w", encoding="utf-8") as f:
    json.dump(ranking, f, ensure_ascii=False, indent=2)
print(f"Wrote {OUT_RANK_JSON} ({len(ranking)} series)")
