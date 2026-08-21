import yt_dlp, os, time, random, pandas as pd, subprocess, sys
from tqdm import tqdm

# === 0. Set working directory ===
os.chdir("/Users/alexg20/Downloads/STA160_project")
print("✅ Working directory set to:", os.getcwd())

# === 1. Cookie validation test ===
TEST_URL = "https://www.youtube.com/watch?v=CD-E-LDc384"

def test_cookies():
    print("\n🔍 Testing cookies from Chrome session...")
    try:
        test_opts = {
            "cookiesfrombrowser": ("chrome",),
            "quiet": True,
            "extract_flat": True,
            "skip_download": True,
            "extractor_args": {"youtube": {"player_client": ["web_embedded"]}},
        }
        with yt_dlp.YoutubeDL(test_opts) as ydl:
            ydl.extract_info(TEST_URL, download=False)
        print("✅ Live Chrome cookies detected — proceeding with downloads.\n")
        return True
    except Exception as e:
        print(f"❌ Cookie test failed: {e}")
        print("⚠️ Make sure you’re logged into YouTube in Chrome and try again.")
        return False

if not test_cookies():
    sys.exit(1)

# === 2. Load dataset ===
df = (
    pd.read_csv("Spotify Youtube Dataset.csv")
    .dropna(subset=["Url_youtube"])
    .drop_duplicates("Url_youtube")
)
df["Artist"] = df["Artist"].str.strip().str.title()
df_sampled = df.groupby("Artist").head(3).reset_index(drop=True)
print(f"🎵 Loaded {len(df_sampled)} songs across {df_sampled['Artist'].nunique()} artists.")

# === 3. Setup folders and checkpointing ===
os.makedirs("audio_data", exist_ok=True)
checkpoint_file = "downloaded_urls.txt"
failed_log = "failed_urls.txt"
sabr_log = "sabr_blocked.txt"
downloaded = (
    set(open(checkpoint_file).read().splitlines())
    if os.path.exists(checkpoint_file)
    else set()
)

# === 4. yt-dlp configuration ===
ydl_opts = {
    "format": "bestaudio[ext=m4a]/bestaudio/best",
    "outtmpl": "audio_data/%(title)s.%(ext)s",
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "opus",
            "preferredquality": "96",
        }
    ],
    "extractor_args": {"youtube": {"player_client": ["web_embedded"]}},
    "cookiesfrombrowser": ("chrome",),
    "http_headers": {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Safari/605.1.15"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.youtube.com/",
        "Origin": "https://www.youtube.com",
    },
    "noplaylist": True,
    "retries": 10,
    "fragment_retries": 10,
    "skip_unavailable_fragments": True,
    "continuedl": True,
    "allow_unplayable_formats": False,
    "quiet": False,
}

batch_size = 100
max_batches_before_pause = 20

# === 5. Batch download loop ===
for start in range(0, len(df_sampled), batch_size):
    batch = df_sampled.iloc[start:start + batch_size]
    batch_num = start // batch_size + 1
    print(f"\n🎵 Starting batch {batch_num} ({len(batch)} songs)...")

    for _, row in tqdm(batch.iterrows(), total=len(batch)):
        url = row["Url_youtube"]
        if url in downloaded:
            continue

        time.sleep(random.uniform(1, 3))  # mimic human behaviour

        try:
            # Step 1 — Get video metadata
            with yt_dlp.YoutubeDL(ydl_opts) as meta_ydl:
                info = meta_ydl.extract_info(url, download=False)
                duration = info.get("duration", None)
                title = info.get("title", "unknown_title").replace("/", "_")

            # Step 2 — Download audio only
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Step 3 — Trim to half duration
            if duration and duration > 0:
                half_time = duration / 2
                input_file = f"audio_data/{title}.opus"
                temp_output = f"audio_data/{title}_trimmed.opus"

                if os.path.exists(input_file):
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        "-i", input_file,
                        "-t", str(half_time),
                        "-c", "copy",
                        temp_output,
                    ]
                    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    os.remove(input_file)
                    os.rename(temp_output, input_file)

            # Step 4 — Save checkpoint
            with open(checkpoint_file, "a") as f:
                f.write(url + "\n")
            downloaded.add(url)
            time.sleep(random.uniform(2, 5))

        except Exception as e:
            error_text = str(e).lower()
            if "403" in error_text or "forbidden" in error_text:
                with open(sabr_log, "a") as f:
                    f.write(url + "\n")
                print(f"🚫 SABR-blocked: {url}")
            else:
                with open(failed_log, "a") as f:
                    f.write(f"{url}\t{str(e)}\n")
                print(f"⚠️ Failed: {url}")

    # === 6. Cooldown + zip batch ===
    print(f"🕒 Finished batch {batch_num}, cooling down before zipping...")
    time.sleep(random.uniform(8, 15))

    output_zip = f"/Users/alexg20/Downloads/STA160_project/audio_batch_{batch_num}.zip"
    subprocess.run(
        ["zip", "-r", output_zip, "audio_data"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    subprocess.run(["rm", "-rf", "audio_data"])
    os.makedirs("audio_data", exist_ok=True)

    print(f"✅ Batch {batch_num} archived and cleared.")

    if batch_num % max_batches_before_pause == 0:
        print("\n🛑 Cookie refresh checkpoint reached!")
        print("👉 Please re-export 'youtube_cookies.txt' and rerun the script to continue.")
        break

