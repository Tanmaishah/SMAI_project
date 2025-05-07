import json
import csv
import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import SSLError, RequestException
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

# Configuration
INPUT_JSON = "Wyscout/players.json"
OUTPUT_CSV = "wyscout_to_transfermarkt_ids.csv"
SEARCH_URL = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9"
}
FUZZY_THRESHOLD = 85  # if you later add fuzzy scoring

# --- Setup a Session with retries ---
session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,                 # waits: 1s, 2s, 4s, 8s, ...
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)


def fetch_search_page(query: str):
    """
    Fetches the Transfermarkt search page for `query`,
    retrying on SSL errors and disabling verification as a last resort.
    """
    url = SEARCH_URL.format(query)
    try:
        return session.get(url, headers=HEADERS, timeout=10)
    except SSLError as e:
        print(f"SSL error for {query!r}: {e}. Retrying without verify.")
        return session.get(url, headers=HEADERS, timeout=10, verify=False)
    except RequestException as e:
        print(f"RequestException for {query!r}: {e}")
        return None


def main():
    # Load Wyscout data
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        wyscout_players = json.load(f)

    # Prepare output CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['wyId', 'name', 'transferMarkt_id'])

        for player in wyscout_players:
            wy_id = player.get('wyId')
            # build full name
            first = player.get('firstName', '').strip()
            last = player.get('lastName', '').strip()
            name = f"{first} {last}".strip()
            if not name:
                continue
            try:
                name = name.encode('utf-8').decode('unicode_escape')
            except Exception:
                # if it fails, just keep the original
                pass

            # perform search
            query = name.replace(' ', '+')
            query = quote_plus(name)
            resp = fetch_search_page(query)
            if not resp or resp.status_code != 200:
                print(f"Failed to fetch for {name!r} (wyId={wy_id})")
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')
            first_row = soup.select_one('table.items tbody tr')
            if not first_row:
                print(f"No results table for {name!r} (wyId={wy_id})")
                continue

            # extract the <a> in td.hauptlink
            link = first_row.select_one('td.hauptlink a')
            if not link or not link.has_attr('href'):
                print(f"No hauptlink <a> for {name!r} (wyId={wy_id})")
                continue

            href = link['href']                      # e.g. '/harun-tekin/profil/spieler/85213'
            tm_id = href.split('/')[-1]              # '85213'
            print(f"{wy_id}\t{name}\t→ tmId={tm_id}")

            writer.writerow([wy_id, name, tm_id])

            # delay to avoid rate‐limit
            time.sleep(random.uniform(1.0, 3.0))

    print(f"Matching complete. Results in {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
