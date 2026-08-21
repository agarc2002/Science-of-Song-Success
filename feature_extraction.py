import os
import pandas as pd
from tqdm import tqdm
from audio_utils import extract_audio_features

def process_audio_files(audio_dir, output_csv="audio_features_full.csv", failed_log="failed_audio_files.txt"):
    # === Load existing CSV to avoid reprocessing ===
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        processed_files = set(existing["filename"])
        print(f"🔁 Found existing {len(processed_files)} processed files.")
    else:
        processed_files = set()
        print("🆕 No existing feature file found — starting fresh.")

    # === Find new files ===
    audio_files = [
        f for f in os.listdir(audio_dir)
        if f.endswith((".m4a", ".opus", ".mp3", ".wav")) and f not in processed_files
    ]
    print(f"🎧 Found {len(audio_files)} new audio files to process.")

    # === Feature extraction ===
    new_features = []
    failed_files = []

    for file in tqdm(audio_files):
        fpath = os.path.join(audio_dir, file)
        result = extract_audio_features(fpath)
        if result:
            new_features.append(result)
        else:
            failed_files.append(file)

    # === Append new features ===
    if new_features:
        df_new = pd.DataFrame(new_features)
        df_combined = pd.concat([existing, df_new], ignore_index=True)
        df_combined.to_csv(output_csv, index=False)
        print(f"✅ Added {len(new_features)} new songs → {output_csv}")
    else:
        print("📁 No new songs to process.")

    # === Log failed files ===
    if failed_files:
        with open(failed_log, "a") as f:
            for name in failed_files:
                f.write(name + "\n")
        print(f"⚠️ Logged {len(failed_files)} failed files → {failed_log}")
    else:
        print("🎉 No failed files this run.")

if __name__ == "__main__":
    audio_dir = "/Users/alexg20/Downloads/STA160_project/audio_data"
    process_audio_files(audio_dir)
