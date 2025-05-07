import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CSV_PATH = "player_cluster_scores.csv"
OUTPUT_DIR = "Cluster_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

for c in sorted(df['cluster'].unique()):
    scores = df.loc[df['cluster'] == c, 'score_1_100'].values

    counts, bins = np.histogram(scores, bins=100, range=(1, 100))
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure()
    plt.plot(bin_centers, counts)
    plt.title(f'Cluster {c} Score Distribution')
    plt.xlabel('Score (1-100)')
    plt.ylabel('Number of Players')
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f'cluster_{c}_distribution.png')
    plt.savefig(output_path)
    plt.close()

print(f"Saved distribution plots for each cluster in '{OUTPUT_DIR}'")
