import yt_dlp, os, time, random, pandas as pd, subprocess, re, zipfile # <-- Add zipfile here
from tqdm import tqdm
from collections import deque

# ===================== USER CONFIG =====================
INPUT_CSV       = r"C:\Users\Yuxiao Tan\OneDrive\桌面\Spotify Youtube Dataset.csv"
SAVE_DIR        = r"F:\aivideo"  # This is where the .opus files will be saved
CHECKPOINT_FILE = r"C:\Users\Yuxiao Tan\OneDrive\桌面\yt_downloaded_urls.txt"
FAILED_LOG      = r"C:\Users\Yuxiao Tan\OneDrive\桌面\yt_failed_urls.txt"
COOKIE_FILE     = r"C:\Users\Yuxiao Tan\OneDrive\桌面\youtube_cookies.txt"
ZIP_SAVE_PATH   = r"F:\aivideo\zipfile" # Where to save the final .zip files

# ❗ IMPORTANT: Update this path to your local ffmpeg.exe
FFMPEG_PATH = r"C:\Users\Yuxiao Tan\OneDrive\桌面\sta160\ffmpeg-2025-10-19-git-dc39a576ad-essentials_build\bin"

# =======================================================

print(f"✅ Saving audio to: {SAVE_DIR}")
print(f"✅ Loading data from: {INPUT_CSV}")

max_batches_before_pause = 20  # ⏸️ pause every 20 batches (~2000 songs)
batch_size = 100

# === 1. Load dataset ===
START_SONG_NAME = "Zach Bryan (feat. Maggie Rogers) - Dawns" # Use the part we're sure about

try:
    df = pd.read_csv(INPUT_CSV)
    
    track_col_name = None
    for col in ["Track","track","Title","title","Name","name"]:
        if col in df.columns:
            track_col_name = col
            break
    
    if not track_col_name:
        print(f"❌ ERROR: Could not find a 'Track' or 'Title' column in your CSV.")
        print(f"   Please check your CSV columns: {df.columns.to_list()}")
        exit()
    
    start_index = 0 
    matches = df[df[track_col_name].fillna('').str.startswith(START_SONG_NAME)]
    
    if not matches.empty:
        start_index = matches.index[0] 
        print(f"✅ Found starting song '{START_SONG_NAME}' at index {start_index}.")
        print(f"   Starting download from this song onwards.")
    else:
        print(f"⚠️ WARNING: Could not find starting song '{START_SONG_NAME}'.")
        print(f"   The script will start from the beginning (index 0).")

    df_sliced = df.iloc[start_index:].copy() 
    df_sliced["Artist"] = df_sliced["Artist"].str.strip().str.title()
    df_to_process = df_sliced.dropna(subset=["Url_youtube"]).drop_duplicates("Url_youtube")
    
    print(f"   Original CSV rows from start song: {len(df_sliced)}")
    print(f"   Total songs to process (after filtering): {len(df_to_process)}")

except FileNotFoundError:
    print(f"❌ ERROR: Cannot find input file at {INPUT_CSV}")
    exit()

# === 2. Setup folders and checkpointing ===
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(ZIP_SAVE_PATH, exist_ok=True) # <-- NEW: Ensure zip directory exists
downloaded = set(open(CHECKPOINT_FILE).read().splitlines()) if os.path.exists(CHECKPOINT_FILE) else set()
print(f"Loaded {len(downloaded)} already-downloaded URLs from checkpoint.")

try:
    urls_in_scope = set(df_to_process['Url_youtube'])
    urls_in_checkpoint = downloaded
    
    urls_to_attempt_set = urls_in_scope - urls_in_checkpoint
    total_to_attempt = len(urls_to_attempt_set)
    
    print(f"   Found {len(urls_in_scope)} total unique songs in scope.")
    print(f"   Found {len(urls_in_checkpoint.intersection(urls_in_scope))} songs already in checkpoint.")
    print(f"➡️ Total songs remaining to download: {total_to_attempt}")
except KeyError:
    print("❌ ERROR: 'Url_youtube' column not found. Please check your CSV.")
    exit()

# === 3. yt-dlp configuration ===
ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": os.path.join(SAVE_DIR, "%(title)s.%(ext)s"),
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "opus",
        "preferredquality": "96",
    }],
    "ffmpeg_location": FFMPEG_PATH,
    "cookiefile": COOKIE_FILE,
    "extractor_args": {
        "youtube": {
            "player_client": ["android", "android_music"],
            "player_skip": []
        }
    },
    "http_headers": {
        "User-Agent": "com.google.android.apps.youtube.music/6.30.51 (Linux; U; Android 14)"
    },
    "noplaylist": True,
    "retries": 3,
    "quiet": False,
}

if not os.path.exists(os.path.join(FFMPEG_PATH, "ffmpeg.exe")):
    print(f"❌ ERROR: 'ffmpeg.exe' not found at the specified FFMPEG_PATH:")
    print(f"   {FFMPEG_PATH}")
    print("   Please update the FFMPEG_PATH variable at the top of the script.")
    exit()


# ⬇️ NEW: === 4. Find last completed batch ===
last_completed_batch = 0
try:
    zip_files = os.listdir(ZIP_SAVE_PATH)
    for f in zip_files:
        # Use regex to find "audio_batch_NUMBER.zip"
        match = re.match(r"audio_batch_(\d+)\.zip", f, re.IGNORECASE)
        if match:
            batch_num = int(match.group(1))
            if batch_num > last_completed_batch:
                last_completed_batch = batch_num
except Exception as e:
    print(f"⚠️ Could not scan zip directory '{ZIP_SAVE_PATH}': {e}")

# MODIFIED: Only print the batch number, don't calculate a skip index
if last_completed_batch > 0:
    print(f"✅ Found {last_completed_batch} completed batches in '{ZIP_SAVE_PATH}'.")
    print(f"   New batches will start numbering from {last_completed_batch + 1}.")
# ⬆️ END NEW


# === 5. Batch download loop === (Renumbered)

session_success_count = 0
session_fail_count = 0
recent_downloads = deque(maxlen=5) 
total_remaining = total_to_attempt

# ⬇️ MODIFIED: The range now starts back at 0
for start in range(0, len(df_to_process), batch_size):
# ⬆️ END MODIFIED

    batch = df_to_process.iloc[start:start + batch_size]
    # ⬇️ MODIFIED: Add the last_completed_batch to the batch_num for the filename
    batch_num = (start // batch_size) + 1 + last_completed_batch
    # ⬆️ END MODIFIED
    
    print(f"\n🎵 Starting batch {batch_num} ({len(batch)} songs)...")

    # ⬇️ DELETED: The "Clear directory" logic was removed from here
    # print(f"🧹 Cleared {SAVE_DIR} for new batch.")
    # ⬆️ END DELETED

    batch_iterator = tqdm(batch.iterrows(), total=len(batch), desc="Starting batch...")
    for _, row in batch_iterator:
    
        url = row["Url_youtube"]
        if url in downloaded:
            continue

        time.sleep(random.uniform(1, 3))
        
        input_file = ""
        temp_output = ""
        title = "unknown_title" 

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True) 
                title = info.get("title", "unknown_title") 
                duration = info.get("duration", None)

            if duration and duration > 0:
                half_time = duration / 2
                input_file = os.path.join(SAVE_DIR, f"{title}.opus")
                temp_output = os.path.join(SAVE_DIR, f"{title}_trimmed.opus")
                
                if not os.path.exists(input_file):
                    raise FileNotFoundError(f"yt-dlp reported download but file not found: {input_file}")

                ffmpeg_cmd = [
                    os.path.join(FFMPEG_PATH, "ffmpeg.exe"),
                    "-y",  # overwrite
                    "-i", input_file,
                    "-t", str(half_time),
                    "-c", "copy",
                    temp_output
                ]
                subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

                os.remove(input_file)
                os.rename(temp_output, input_file)

            with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
                f.write(url + "\n")
            downloaded.add(url)
            
            session_success_count += 1
            recent_downloads.append(title)
            total_remaining = total_to_attempt - session_success_count

            time.sleep(random.uniform(2, 5))

        except Exception as e:
            err_str = str(e) 
            with open(FAILED_LOG, "a", encoding="utf-8") as f:
                f.write(f"{url}\t{err_str}\n")
            print(f"⚠️ Failed: {url} | Error: {err_str[:200]}...") 
            
            session_fail_count += 1 

            if "captcha challenge" in err_str:
                print("\n" + "="*70)
                print("🛑 CRITICAL: CAPTCHA required. YouTube has blocked this session.")
                print(f"👉 Please re-export '{COOKIE_FILE}' (refresh your login) and restart.")
                print("   Stopping all further downloads to avoid account/IP lock.")
                print("="*70)
                raise SystemExit("CAPTCHA block detected.") 
            
            try:
                if input_file and os.path.exists(input_file):
                    os.remove(input_file)
                if temp_output and os.path.exists(temp_output):
                    os.remove(temp_output)
            except Exception as clean_e:
                print(f"  (Cleanup failed: {clean_e})")
        
        batch_iterator.set_description(
            f"Success: {session_success_count} | Failed: {session_fail_count} | Remaining: {total_remaining}"
        )

    
# === 6. Cool-down and zip ===
print(f"🕒 Finished batch {batch_num}, cooling down before zipping...")
print(f"   Last 5 successful downloads: {list(recent_downloads)}")
time.sleep(random.uniform(8, 15))

output_zip = os.path.join(ZIP_SAVE_PATH, f"audio_batch_{batch_num}.zip")
tmp_zip    = os.path.join(ZIP_SAVE_PATH, f"audio_batch_{batch_num}.zip.tmp")

# 1) Gather files to zip
files_to_zip = [
    f for f in os.listdir(SAVE_DIR)
    if os.path.isfile(os.path.join(SAVE_DIR, f))
    and f.lower().endswith((".opus", ".mp3", ".m4a"))
]

if not files_to_zip:
    print(f"⚠️ No files found in {SAVE_DIR}. Skipping zip for batch {batch_num}.")
else:
    # 2) Do ONLY the zip in this try/except
    try:
        print(f"   Found {len(files_to_zip)} files to zip... (using Python zipfile)")
        # write to temp file first; allowZip64 for large archives
        with zipfile.ZipFile(tmp_zip, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for f in files_to_zip:
                file_path = os.path.join(SAVE_DIR, f)
                zf.write(file_path, arcname=f)

        # ensure the file exists and is non-empty before replacing
        if not os.path.exists(tmp_zip) or os.path.getsize(tmp_zip) == 0:
            raise RuntimeError("Temp zip not created or empty.")

        # atomic move to final name
        os.replace(tmp_zip, output_zip)
        print(f"✅ Batch {batch_num} archived to {output_zip}")

    except Exception as zip_e:
        # If zip creation failed, clean up only the temp zip
        try:
            if os.path.exists(tmp_zip):
                os.remove(tmp_zip)
        except Exception:
            pass
        print(f"❌ Failed to zip batch {batch_num}. Error: {zip_e}")
        print("   Leaving audio files in place so you can retry next run.")
    else:
        # 3) Cleanup is separate; never delete the finished zip on cleanup errors
        print(f"🧹 Clearing {SAVE_DIR} for next batch...")
        try:
            for f in os.listdir(SAVE_DIR):
                p = os.path.join(SAVE_DIR, f)
                if os.path.isfile(p) and f.lower().endswith((".opus", ".part", ".ytdl", ".m4a", ".mp3")):
                    try:
                        os.remove(p)
                    except PermissionError as pe:
                        print(f"  ⚠️ Skipped (in use): {p} ({pe})")
                    except Exception as e:
                        print(f"  ⚠️ Skipped: {p} ({e})")
        except Exception as cleanup_e:
            print(f"  ⚠️ Cleanup had issues: {cleanup_e}")


print("\n🎉 All batches complete.")