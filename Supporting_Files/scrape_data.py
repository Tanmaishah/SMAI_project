import requests
import csv
from datetime import datetime
import time
import random

# --- Step 1: Load player IDs from matched_player_values.csv ---
player_ids = set()

with open("wyscout_to_transfermarkt_ids.csv", mode="r", encoding="utf-8") as infile:
    reader = csv.reader(infile)
    for row in reader:
        if len(row) >= 2:
            try:
                player_id = int(row[2])
                player_ids.add(player_id)
            except ValueError:
                continue  # Skip rows with invalid IDs

# --- Step 2: Setup for API scraping and output ---
TARGET_YEARS = {2017, 2018}
output_file = "transfermarkt_valuations_2017_2018.csv"

with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["player_id", "date", "market_value_eur", "club", "age"])

    for pid in player_ids:
        url = f"https://www.transfermarkt.co.uk/ceapi/marketValueDevelopment/graph/{pid}"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": f"https://www.transfermarkt.co.uk/player/profil/spieler/{pid}"
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            for entry in data.get("list", []):
                date_str = entry.get("datum_mw", "")
                try:
                    date_obj = datetime.strptime(date_str, "%b %d, %Y")
                    if date_obj.year in TARGET_YEARS:
                        writer.writerow([
                            pid,
                            date_obj.strftime("%Y-%m-%d"),
                            entry.get("y", ""),
                            entry.get("verein", ""),
                            entry.get("age", "")
                        ])
                except ValueError:
                    continue  # skip invalid or missing dates

        except Exception as e:
            print(f"Error for player {pid}: {e}")
    time.sleep(random.uniform(1.5, 3.5))
