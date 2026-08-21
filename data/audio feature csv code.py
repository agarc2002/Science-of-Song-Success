import os
import time
import gc
import psutil
import pandas as pd
from tqdm import tqdm
import subprocess
import multiprocessing as mp
from audio_utils import extract_audio_features

# === CONFIG ===
AUDIO_DIR = "/Users/alexg20/Downloads/STA160_project/audio_data"
OUTPUT_CSV = "/Users/alexg20/Downloads/STA160_project/audio_features_full_all_20251025_1205.csv"
FAILED_LOG = "/Users/alexg20/Downloads/STA160_project/failed_audio_files.txt"
SAVE_INTERVAL = 100
MEMORY_LIMIT_MB = 10000  # 10 GB (process memory)
TIMEOUT_SEC = 60  # Kill after 60 seconds

# === Worker function must be top-level for macOS ===
def worker_extract(q, file_path):
    """Runs feature extraction and returns result or exception."""
    try:
        res = extract_audio_features(file_path)
        q.put(res)
    except Exception as e:
        q.put(e)

# === Safe feature extraction wrapper with timeout ===
def extract_with_timeout(file_path, timeout=TIMEOUT_SEC):
    """Run extract_audio_features in a subprocess with timeout protection."""
    q = mp.Queue()
    p = mp.Process(target=worker_extract, args=(q, file_path))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return TimeoutError(f"Timed out after {timeout}s")

    if not q.empty():
        result = q.get()
        if isinstance(result, Exception):
            return result
        return result
    return None


# === MAIN PROGRAM ===
if __name__ == "__main__":
    mp.freeze_support()  # required on macOS and Windows

    # === 1. Gather audio files ===
    all_files = [
        os.path.join(AUDIO_DIR, f)
        for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith((".opus", ".mp3", ".wav", ".m4a"))
    ]
    total_files = len(all_files)
    print(f"🎧 Found {total_files} total audio files.")

    # === 2. Resume from existing CSV ===
    processed = set()
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        if "filename" in existing.columns:
            processed = set(existing["filename"].dropna())
        print(f"🔁 Found {len(processed)} processed files — skipping them.")
    else:
        existing = pd.DataFrame()
        print("🆕 No existing CSV found — starting from scratch.")

    remaining_files = [f for f in all_files if os.path.basename(f) not in processed]
    print(f"➡️ {len(remaining_files)} files left to process.\n")

    # === 3. Process audio files ===
    start_time = time.time()
    batch_results = []
    failed_files = []

    progress_bar = tqdm(total=len(remaining_files), desc="🎵 Extracting features", unit="file", dynamic_ncols=True, leave=False)

    for i, file in enumerate(remaining_files, start=1):
        basename = os.path.basename(file)
        progress_bar.write(f"🎧 Processing: {basename}")

        result = extract_with_timeout(file, timeout=TIMEOUT_SEC)

        if isinstance(result, TimeoutError):
            progress_bar.write(f"⏱️ Timeout on {basename} ({TIMEOUT_SEC}s) — skipped.")
            failed_files.append(file)
        elif isinstance(result, Exception):
            progress_bar.write(f"⚠️ Error processing {basename}: {result}")
            failed_files.append(file)
        elif result is None:
            progress_bar.write(f"🚫 No features extracted for {basename}")
            failed_files.append(file)
        else:
            batch_results.append(result)

        # Update progress bar
        processed_total = len(processed) + i
        progress_bar.set_postfix_str(f"{processed_total}/{total_files} processed")
        progress_bar.update(1)

        # === Auto-save early if memory high (process only) ===
        process = psutil.Process(os.getpid())
        mem_use = process.memory_info().rss / (1024 ** 2)
        if mem_use > MEMORY_LIMIT_MB:
            progress_bar.write(f"⚠️ High Python memory usage ({mem_use:.0f} MB) — saving early...")
            if batch_results:
                df_new = pd.DataFrame(batch_results)
                df_new.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
                batch_results = []
            gc.collect()

        # === Periodic autosave ===
        if i % SAVE_INTERVAL == 0 or i == len(remaining_files):
            if batch_results:
                df_new = pd.DataFrame(batch_results)
                df_new.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
                batch_results = []
            elapsed = (time.time() - start_time) / 60
            remaining = len(remaining_files) - i
            progress_bar.write(f"💾 Autosaved after {i} files ({remaining} remaining). Elapsed: {elapsed:.1f} min")
            gc.collect()

    progress_bar.close()

    # === 4. Save failed files ===
    if failed_files:
        with open(FAILED_LOG, "a") as f:
            for name in failed_files:
                f.write(name + "\n")
        print(f"⚠️ Logged {len(failed_files)} failed files → {FAILED_LOG}")

    # === 5. Summary ===
    elapsed_total = (time.time() - start_time) / 60
    print("\n✅ Feature extraction complete!")
    print(f"🕒 Total time: {elapsed_total:.1f} minutes")
    print(f"💽 Output CSV: {OUTPUT_CSV}")
