import pandas as pd
import pickle
from collections import defaultdict
from datetime import datetime

cluster_df = pd.read_csv('player_cluster_scores.csv')  
with open('Models_XGBoost/cluster_map.pkl', 'rb') as f:
    cluster_map = pickle.load(f)

player_matches = defaultdict(lambda: defaultdict(set))

for cluster_id, matches in cluster_map.items():
    for match_id, teams in matches.items():
        for team_id, players in teams.items():
            for player_id in players.keys():
                player_matches[player_id][cluster_id].add(match_id)

player_cluster_counts = {
    player_id: {
        cluster_id: len(match_set)
        for cluster_id, match_set in clusters.items()
    }
    for player_id, clusters in player_matches.items()
}

primary_cluster_map = {}
for player_id, clusters in player_cluster_counts.items():
    primary_cluster = max(clusters.items(), key=lambda item: (item[1], -item[0]))[0]
    primary_cluster_map[player_id] = primary_cluster

df = pd.read_csv('transfermarkt_valuations_2017_2018.csv', parse_dates=['date'])

player_map = defaultdict(lambda: {'market_value_2017': [], 'market_value_2018': [], 'age': None})

for _, row in df.iterrows():
    pid = row['player_id']
    year = row['date'].year
    value = row['market_value_eur']
    
    if year == 2017:
        player_map[pid]['market_value_2017'].append(value)
    elif year == 2018:
        player_map[pid]['market_value_2018'].append(value)
    
    player_map[pid]['age'] = row['age']

tm_market_values = {}
for pid, data in player_map.items():
    avg_2017 = sum(data['market_value_2017']) / len(data['market_value_2017']) if data['market_value_2017'] else None
    avg_2018 = sum(data['market_value_2018']) / len(data['market_value_2018']) if data['market_value_2018'] else None
    tm_market_values[pid] = {
        'market_value_2017': avg_2017,
        'market_value_2018': avg_2018,
        'age': data['age']
    }

tm_to_ws_df = pd.read_csv('wyscout_to_transfermarkt_ids.csv')  
tm_to_ws_map = dict(zip(tm_to_ws_df['transferMarkt_id'], tm_to_ws_df['wyId']))

ws_score_map = {
        row['player_id']: {
            'cluster': row['cluster'],
            'score': row['score_1_100']
        }
    for _, row in cluster_df.iterrows()
    if row['cluster'] == primary_cluster_map.get(row['player_id'])
}

final_map = {}

for tm_id, market_data in tm_market_values.items():
    if tm_id in tm_to_ws_map:
        ws_id = tm_to_ws_map[tm_id]

        entry = {
            'market_value_2017': market_data.get('market_value_2017'),
            'market_value_2018': market_data.get('market_value_2018'),
            'age': market_data.get('age'),
        }

        if ws_id in ws_score_map:
            entry['cluster'] = ws_score_map[ws_id]['cluster']
            entry['score'] = ws_score_map[ws_id]['score']
        else:
            entry['cluster'] = 1
            entry['score'] = 1.0
        if entry['market_value_2018'] and entry['market_value_2017']:
            final_map[ws_id] = entry

df = pd.DataFrame.from_dict(final_map, orient='index')

df['mv17_m'] = df['market_value_2017'] / 1e6
df['mv18_m'] = df['market_value_2018'] / 1e6

feature_cols = ['mv17_m', 'score', 'age', 'cluster']
X = df[feature_cols].to_numpy()
y = df['mv18_m'].to_numpy()

with open('Prediction_data_RF/player_valuation_data.pkl', 'wb') as f:
    pickle.dump({'X': X, 'y': y}, f)