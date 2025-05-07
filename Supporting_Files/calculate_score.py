import os
import pickle
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from xgboost import XGBClassifier 

# --- config -----------------------------
MODELS_DIR   = "Models_XGBoost"
OUTPUT_CSV   = "player_cluster_scores.csv"
NUM_CLUSTERS = 8

# dict: cluster_map[c][match_id][team_id][player_id] = {feat: val, ...}
with open(os.path.join(MODELS_DIR,"cluster_map.pkl"), "rb") as f:
    cluster_map = pickle.load(f)

all_rows = []

for c in range(1, NUM_CLUSTERS+1):
    # to load the trained model and feature order for this cluster
    model_path = os.path.join(MODELS_DIR, f"xgb_cluster{c}.pkl")
    feat_path  = os.path.join(MODELS_DIR, f"featnames_cluster{c}.pkl")
    if not (os.path.exists(model_path) and os.path.exists(feat_path)):
        print(f"[Cluster {c}] model or featnames not found, skipping.")
        continue

    model: XGBClassifier = joblib.load(model_path)
    feature_names = pickle.load(open(feat_path, "rb"))
    importances = model.feature_importances_

    # get player's average performance across the season
    player_match_feats = defaultdict(list)
    for match_id, teams in cluster_map[c].items():
        for team_id, players in teams.items():
            for pid, feats in players.items():
                # ensure vector in the right order
                vec = np.array([feats[f] for f in feature_names], dtype=float)
                player_match_feats[pid].append(vec)

    raw_scores = {}
    for pid, vecs in player_match_feats.items():
        avg_vec = np.mean(vecs, axis=0)        
        S_p = float(np.dot(importances, avg_vec))
        raw_scores[pid] = S_p

    # normalize raw scores between 1–100
    scores = np.array(list(raw_scores.values()))
    min_s, max_s = scores.min(), scores.max()
    rng = max_s - min_s if max_s>min_s else 1.0

    for pid, S_p in raw_scores.items():
        norm = 1 + 99 * (S_p - min_s) / rng
        all_rows.append({
            "cluster": c,
            "player_id": pid,
            "raw_score": S_p,
            "score_1_100": norm
        })

df = pd.DataFrame(all_rows).sort_values(["cluster","score_1_100"], ascending=[True,False])
df.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote {len(df)} player scores to {OUTPUT_CSV}")
