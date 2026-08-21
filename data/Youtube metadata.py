from googleapiclient.discovery import build
import pandas as pd
from tqdm import tqdm
import time
import os

# === 1. Load dataset ===
df = pd.read_csv("Spotify Youtube Dataset.csv")

# === 2. Clean dataset ===
df = (
    df.dropna(subset=["Url_youtube"])
      .drop_duplicates("Url_youtube")
      .reset_index(drop=True)
)
print(f"🎵 Loaded {len(df)} songs across {df['Artist'].nunique()} artists")

# === 3. Setup YouTube API ===
API_KEY = "Insert API Key"  # 🔑 replace with your key
youtube = build("youtube", "v3", developerKey=API_KEY)

# === 4. Extract video IDs ===
df["video_id"] = df["Url_youtube"].str.extract(r"v=([a-zA-Z0-9_-]{11})")
video_ids = df["video_id"].dropna().unique().tolist()

# === 5. Load existing checkpoint (if any) ===
checkpoint_file = "youtube_stats_checkpoint.csv"
if os.path.exists(checkpoint_file):
    stats_df = pd.read_csv(checkpoint_file)
    done_videos = set(stats_df["video_id"].tolist())
    print(f"🔁 Resuming from checkpoint ({len(done_videos)} videos already processed)")
else:
    stats_df = pd.DataFrame(columns=["video_id", "Views", "Likes", "Comments"])
    done_videos = set()

# === 6. Helper to fetch stats for batches ===
def fetch_video_stats_batch(ids):
    try:
        request = youtube.videos().list(part="statistics", id=",".join(ids[:50]))
        response = request.execute()
        stats_list = []
        for item in response.get("items", []):
            vid = item["id"]
            s = item.get("statistics", {})
            stats_list.append({
                "video_id": vid,
                "Views": int(s.get("viewCount", 0)),
                "Likes": int(s.get("likeCount", 0)),
                "Comments": int(s.get("commentCount", 0))
            })
        return stats_list
    except Exception as e:
        print("⚠️ API batch failed:", e)
        return []

# === 7. Loop through all videos, resuming from checkpoint ===
stats_all = stats_df.to_dict("records")
start_idx = len(done_videos)

for i in tqdm(range(start_idx, len(video_ids), 50), desc="Fetching YouTube stats"):
    batch = [vid for vid in video_ids[i:i+50] if vid not in done_videos]
    if not batch:
        continue

    new_stats = fetch_video_stats_batch(batch)
    if new_stats:
        stats_all.extend(new_stats)
        done_videos.update([s["video_id"] for s in new_stats])

    # 💾 Save checkpoint every 500 videos
    if i % 500 == 0 or i + 50 >= len(video_ids):
        pd.DataFrame(stats_all).to_csv(checkpoint_file, index=False)
        print(f"💾 Checkpoint saved: {len(done_videos)} videos processed")
    time.sleep(0.1)

# === 8. Final merge ===
stats_df = pd.DataFrame(stats_all).drop_duplicates("video_id")
print(f"\n✅ Total stats collected: {len(stats_df)}")

df_updated = df.merge(stats_df, on="video_id", how="left")

# === 9. Fix column naming (handle _x / _y conflicts) ===
for metric in ["Views", "Likes", "Comments"]:
    if f"{metric}_y" in df_updated.columns:
        df_updated[metric] = df_updated[f"{metric}_y"]
    elif f"{metric}_x" in df_updated.columns:
        df_updated[metric] = df_updated[f"{metric}_x"]
    else:
        df_updated[metric] = None

# === 10. Drop audio columns, keep all others + updated stats ===
audio_cols = [
    "Danceability", "Energy", "Key", "Loudness", "Speechiness",
    "Acousticness", "Instrumentalness", "Liveness", "Valence",
    "Tempo", "Duration_ms"
]

df_final = df_updated.drop(columns=[col for col in audio_cols if col in df_updated.columns], errors="ignore")

# Drop 'Description' column if it exists
if "Description" in df_final.columns:
    df_final = df_final.drop(columns=["Description"])

# === 11. Create and save the final clean non-audio + updated stats CSV ===

# Drop Spotify audio-related columns
audio_cols = [
    "Danceability", "Energy", "Key", "Loudness", "Speechiness",
    "Acousticness", "Instrumentalness", "Liveness", "Valence",
    "Tempo", "Duration_ms"
]
df_final = df_updated.drop(columns=[col for col in audio_cols if col in df_updated.columns], errors="ignore")

# Fix YouTube stats naming
for metric in ["Views", "Likes", "Comments"]:
    if f"{metric}_y" in df_final.columns:
        df_final[metric] = df_final[f"{metric}_y"]
    elif f"{metric}_x" in df_final.columns:
        df_final[metric] = df_final[f"{metric}_x"]

# Remove duplicates, old index, Description, and Stream
drop_cols = [
    col for col in df_final.columns
    if col.endswith(('_x', '_y')) or col in ["Unnamed: 0", "Description", "Stream"]
]
df_final = df_final.drop(columns=drop_cols, errors="ignore")

# === Save clean final CSV ===
output_path = os.path.join(os.getcwd(), "Spotify_Youtube_NonAudio_Clean.csv")
df_final.to_csv(output_path, index=False)

print("\n💾 Saved clean unified dataset for group (no Description or Stream columns):")
print(output_path)
print(f"Rows: {len(df_final)}, Columns: {len(df_final.columns)}")
print("\nKept columns:")
print(df_final.columns.tolist())
print("\nPreview:")
print(df_final.head(3))


