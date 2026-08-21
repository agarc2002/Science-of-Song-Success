import pandas as pd
import time
from tqdm import tqdm
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.client import Spotify
import os
import sys

# === 0. Set working directory ===
os.chdir("/Users/alexg20/Downloads/STA160_project")
print("📂 Working directory set to:", os.getcwd())

# === 1. Setup Spotify API with timeout ===
auth_manager = SpotifyClientCredentials(
    client_id="Insert Client ID",        # ✅ Your actual Spotify Client ID
    client_secret="Insert Client Secret"     # ✅ Your actual Spotify Client Secret
)
sp = Spotify(auth_manager=auth_manager, requests_timeout=10, retries=3)

# === 2. Load dataset ===
df = pd.read_csv("Youtube_MegaData.csv")   # Make sure this file exists in STA160_project
print(f"🎧 Loaded {len(df)} songs from {df['Artist'].nunique()} artists")

# === 3. Checkpoint setup ===
checkpoint_file = "spotify_tracklinks_checkpoint.csv"
if os.path.exists(checkpoint_file):
    checkpoint_df = pd.read_csv(checkpoint_file)
    done_songs = set(checkpoint_df["Track"] + checkpoint_df["Artist"])
    print(f"🔁 Resuming from checkpoint — {len(done_songs)} songs already processed.")
else:
    checkpoint_df = pd.DataFrame(columns=["Artist", "Track", "Track_URL_Spotify", "Spotify_Track_Popularity"])
    done_songs = set()

# === 4. Helper function ===
def get_spotify_info(artist, track):
    try:
        query = f"track:{track} artist:{artist}"
        result = sp.search(q=query, type="track", limit=1)
        items = result.get("tracks", {}).get("items", [])
        if items:
            track_info = items[0]
            url = track_info["external_urls"]["spotify"]
            popularity = track_info["popularity"]
            return url, popularity
        return None, None
    except Exception as e:
        print(f"⚠️ Error fetching {artist} - {track}: {e}")
        return None, None

# === 5. Quick test to confirm Spotify API works ===
print("🔍 Testing Spotify API on sample song...")
test_url, test_pop = get_spotify_info("Gorillaz", "Feel Good Inc.")
print("Test result:", test_url, test_pop)
if not test_url:
    print("❌ Spotify API test failed — check credentials or network connection.")
    sys.exit(1)

# === 6. Process songs in batches ===
batch_size = 1000

for i in tqdm(range(0, len(df), batch_size), desc="Fetching Spotify track data"):
    batch = df.iloc[i:i + batch_size]
    print(f"🌀 Starting batch {i//batch_size + 1}, {len(batch)} songs to process...")
    sys.stdout.flush()
    batch_records = []

    for j, row in enumerate(batch.itertuples(index=False)):
        key = row.Track + row.Artist
        if key in done_songs:
            continue

        try:
            url, popularity = get_spotify_info(row.Artist, row.Track)
        except Exception as e:
            print(f"⚠️ Error fetching {row.Artist} - {row.Track}: {e}")
            url, popularity = None, None

        batch_records.append({
            "Artist": row.Artist,
            "Track": row.Track,
            "Track_URL_Spotify": url,
            "Spotify_Track_Popularity": popularity
        })
        done_songs.add(key)

        # 👇 Live progress update every 25 songs
        if (j + 1) % 25 == 0:
            print(f"  🎵 Processed {j + 1} songs in current batch")
            sys.stdout.flush()

        time.sleep(0.1)  # Respect Spotify API rate limits

    # 💾 Save after each batch
    if batch_records:
        new_df = pd.DataFrame(batch_records)
        checkpoint_df = pd.concat([checkpoint_df, new_df], ignore_index=True)
        checkpoint_df.to_csv(checkpoint_file, index=False)
        print(f"💾 Saved checkpoint — total processed: {len(checkpoint_df)}")
        sys.stdout.flush()

print(f"\n✅ All batches processed. Total songs fetched: {len(checkpoint_df)}")

# === 7. Merge Spotify data into main dataset ===
merged = df.merge(checkpoint_df, on=["Artist", "Track"], how="left")

# === 8. Save final merged output ===
final_path = os.path.join(os.getcwd(), "Spotify_Youtube_WithTrackLinks.csv")
merged.to_csv(final_path, index=False)

print(f"\n💾 Final file saved: {final_path}")
print(f"Rows: {len(merged)}, Columns: {len(merged.columns)}")
print("✅ Done.")





