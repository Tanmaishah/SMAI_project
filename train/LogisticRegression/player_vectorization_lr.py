from collections import defaultdict
import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from player_vectorization_helper import make_stat_map, normalize_new_map,\
                                        process_event, calculate_additional_stats, \
                                        add_normalized_positions, assign_position_clusters, \
                                        get_subs_map, get_team_events, transform_stat_map, \
                                        split_feature_map_by_cluster, plot_all_normalized_clusters

# -----------------File Handling----------------------

events_dir = "Wyscout/events/"
matches_dir = "Wyscout/matches/"

# --------------------- Variables ----------------------------

if __name__ == "__main__":
    stat_map = make_stat_map()
    subs_map, goals_map, opp_goals_map = get_subs_map()
    team_event_times = defaultdict(lambda: defaultdict(list))
    winner_map = {}

    for fn in os.listdir(events_dir):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(events_dir, fn)) as f:
            ev_list = json.load(f)
        for ev in ev_list:
            m = ev["matchId"]
            t = ev["teamId"]
            ts = ev["eventSec"]
            team_event_times[m][t].append(ts)
            process_event(ev, stat_map)

        with open(os.path.join(matches_dir, fn)) as f:
            matches = json.load(f)
        local_winner_map = { m["wyId"]: m.get("winner", 0) for m in matches }
        winner_map.update(local_winner_map)

    calculate_additional_stats(stat_map, goals_map, opp_goals_map)
    add_normalized_positions(stat_map)
    assign_position_clusters(stat_map, n_clusters=8)
    get_team_events(stat_map, subs_map, team_event_times)

    # plot_all_normalized_clusters(stat_map)

    un_feature_map = transform_stat_map(stat_map)
    n_feature_map = normalize_new_map(un_feature_map) 

    cluster_map = split_feature_map_by_cluster(stat_map, n_feature_map, num_clusters=8)
    # for key, value in cluster_maps[2][2500098][1633][8561].items():
    #     print(key, " - ", value)

    os.makedirs("Models_LR", exist_ok=True)
    with open("Models_LR/cluster_map.pkl", "wb") as f:
        pickle.dump(cluster_map, f)

    if isinstance(matches, dict):
        matches = [matches]    

    sample_cluster = next(c for c, cmap in cluster_map.items() if cmap)
    sample_match   = next(m for m in cluster_map[sample_cluster] if cluster_map[sample_cluster][m])
    sample_team    = next(t for t in cluster_map[sample_cluster][sample_match])
    sample_player  = next(iter(cluster_map[sample_cluster][sample_match][sample_team]))
    feature_names  = sorted(cluster_map[sample_cluster][sample_match][sample_team][sample_player].keys())

    for c in range(1, 9):
        X_rows, y_labels = [], []

        for mid, teams in tqdm(cluster_map[c].items(), desc=f"Cluster {c}"):
            win_tid = winner_map.get(mid, 0)
            if win_tid == 0:
                continue
            for tid, players in teams.items():
                vec = [ np.mean([pstats[feat] for pstats in players.values()]) 
                        for feat in feature_names ]
                X_rows.append(vec)
                y_labels.append(1 if tid == win_tid else 0)

        Xc = pd.DataFrame(X_rows, columns=feature_names)
        yc = np.array(y_labels)

        if len(yc)==0:
            continue


        X_train, X_temp, y_train, y_temp = train_test_split(
            Xc, yc, test_size=0.4, random_state=42, stratify=yc
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        joblib.dump(X_train, f"Models/X_train{c}.pkl")
        joblib.dump(y_train, f"Models/y_train{c}pkl")
        joblib.dump(X_val,   f"Models/X_val{c}.pkl")
        joblib.dump(y_val,   f"Models/y_val{c}.pkl")
        joblib.dump(X_test,  f"Models/X_test{c}.pkl")
        joblib.dump(y_test,  f"Models/y_test{c}.pkl")

        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train, y_train)

        joblib.dump(model, f"Models/lr_cluster{c}.pkl")
        with open(f"Models/featnames_cluster{c}.pkl", "wb") as f:
            pickle.dump(feature_names, f)

        print(f"Validation accuracy - {c}: {model.score(X_val, y_val):.3f}")

        coef = pd.Series(model.coef_[0], index=feature_names).sort_values(ascending=False)
        print("-----Model Prediction-----")
        print(coef.to_dict())
        print("-----BaseLine-----")
        print("Class balance:", Counter(y_train))
        print("Majority-class baseline:", max(Counter(y_train).values())/len(y_train))
        print("-"*40)


