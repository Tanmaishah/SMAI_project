import json
import pandas as pd

# Paths to your files
JSON_PATH = "Wyscout/players.json"
CSV_PATH  = "TransferMarkt/players.csv"
OUT_PATH  = "matched_player_values.csv"

with open(JSON_PATH, "r") as f:
    players_json = json.load(f)

records = []
for p in players_json:
    # build token sets for first and last name
    fn_tokens = set(str(p["firstName"]).lower().split())
    ln_tokens = set(str(p["lastName"]).lower().split())
    records.append({
        "player_id_json": p["wyId"],
        "fn_tokens": fn_tokens,
        "ln_tokens": ln_tokens
    })
df_json = pd.DataFrame(records)

# 2) Load and preprocess CSV
df_csv = pd.read_csv(CSV_PATH, usecols=["player_id", "name", "market_value_in_eur"])
# split CSV name into tokens once
df_csv["name_tokens"] = df_csv["name"].str.lower().str.split().apply(set)

print("created csv name set")

# 3) Perform token-based matching
matches = []
for _, j in df_json.iterrows():
    for _, c in df_csv.iterrows():
        # require at least one token from first & last name to appear in the CSV name
        if (j["fn_tokens"] & c["name_tokens"]) and (j["ln_tokens"] & c["name_tokens"]):
            matches.append({
                "player_id_json": j["player_id_json"],
                "player_id_csv":  c["player_id"],
                "name":           c["name"],
                "market_value_in_eur": c["market_value_in_eur"]
            })
            print(c["player_id"], c["name"])

# 4) Save results
df_out = pd.DataFrame(matches)
df_out.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(df_out)} matched rows to {OUT_PATH}")