import os
import re
import time
import json
import urllib.parse
from difflib import get_close_matches

import pandas as pd
import numpy as np
import lyricsgenius
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi

# =====================================================================
# PATHS
# =====================================================================
csv_path = r"C:\Users\rohan\Downloads\STA 160\all_with_metadata_embeddings.csv"
lyrics_checkpoint_path = r"C:\Users\rohan\Downloads\STA 160\lyrics_checkpoint.csv"
embed_checkpoint_path = r"C:\Users\rohan\Downloads\STA 160\embedding_checkpoint.npy"
output_path = r"C:\Users\rohan\Downloads\STA 160\all_with_metadata_embeddings_with_lyrics.csv"

# =====================================================================
# LOAD DATA
# =====================================================================
meta = pd.read_csv(csv_path)
print("Loaded dataset:", meta.shape)

# =====================================================================
# API CLIENTS (INSERT YOUR KEYS HERE)
# =====================================================================
GENIUS_API_KEY = "Insert Key Here"
OPENAI_API_KEY = "Insert Key Here"

genius = lyricsgenius.Genius(
    GENIUS_API_KEY,
    timeout=12,
    skip_non_songs=True,
    retries=3
)

client = OpenAI(api_key=OPENAI_API_KEY)

# =====================================================================
# TITLE CLEANING
# =====================================================================
def clean_title(title):
    if pd.isna(title):
        return ""
    t = str(title)

    t = re.sub(r"\(.*?\)", "", t)
    t = re.sub(r"\[.*?\]", "", t)

    junk_words = [
        "official video", "official music video", "audio", "lyrics",
        "lyric video", "hd", "4k", "remastered", "explicit", "clean",
        "prod.", "feat.", "ft.", "visualizer", "mv"
    ]
    t_low = t.lower()
    for word in junk_words:
        t_low = t_low.replace(word, "")

    t_low = re.sub(r"[^\w\s'-]", "", t_low)
    return " ".join(t_low.split()).strip()

# =====================================================================
# MULTILINGUAL TITLE GENERATION
# =====================================================================
def make_multilingual_titles(title):
    if pd.isna(title) or len(title.strip()) == 0:
        return []

    prompt = (
        f"Translate the song title '{title}' into "
        f"Hindi, Spanish, Portuguese, Arabic, Japanese, Korean, French. "
        f"Return ONLY a JSON list of translated titles."
    )

    try:
        res = client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )
        txt = res.output_text.strip()

        translations = json.loads(txt)
        return [clean_title(t) for t in translations]
    except:
        return []

# =====================================================================
# YOUTUBE TRANSCRIPT FALLBACK
# =====================================================================
def extract_youtube_id(url):
    if pd.isna(url):
        return None
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.hostname in ["youtube.com", "www.youtube.com"]:
            return urllib.parse.parse_qs(parsed.query).get("v", [None])[0]
        if parsed.hostname in ["youtu.be", "www.youtu.be"]:
            return parsed.path[1:]
    except:
        return None
    return None

def get_youtube_transcript(url):
    vid = extract_youtube_id(url)
    if not vid:
        return ""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(
            vid,
            languages=["en", "hi", "es", "pt", "ar", "ja", "ko"]
        )
        return " ".join(seg["text"] for seg in transcript)
    except:
        return ""

# =====================================================================
# ADVANCED MULTILINGUAL GENIUS SEARCH
# =====================================================================
def search_song_multilingual(title, artist):
    clean_t = clean_title(title)
    candidates = [clean_t, title]

    multilingual = make_multilingual_titles(clean_t)
    candidates.extend(multilingual)

    # try all titles with and without artist
    for t in candidates:
        if not t:
            continue
        try:
            s = genius.search_song(t, artist)
            if s:
                return s
        except:
            pass
        try:
            s = genius.search_song(t)
            if s:
                return s
        except:
            pass

    # fuzzy fallback
    try:
        results = genius.search(clean_t, per_page=10, page=1).get("hits", [])
        titles = [r["result"]["full_title"] for r in results]
        best = get_close_matches(clean_t, titles, n=1, cutoff=0.45)
        if best:
            for r in results:
                if r["result"]["full_title"] == best[0]:
                    return genius.song(r["result"]["id"])
    except:
        pass

    return None

# =====================================================================
# PART 1 — LYRICS WITH CHECKPOINTS (SAFE + MULTILINGUAL)
# =====================================================================

if os.path.exists(lyrics_checkpoint_path):
    print("Resuming from lyrics checkpoint...")
    meta = pd.read_csv(lyrics_checkpoint_path)
else:
    meta["Lyrics"] = ""

start_time = time.time()

for i, row in meta.iterrows():

    # Skip if lyrics already obtained
    if isinstance(row["Lyrics"], str) and len(row["Lyrics"]) > 3:
        continue

    artist = str(row["Artist"])
    title = str(row["Track"])
    yt_url = row.get("Url_youtube", "")

    print(f"\n🌍 [{i}] Searching for: \"{title}\" by {artist}")

    # 1️⃣ Genius multilingual search
    song = search_song_multilingual(title, artist)

    # --------------------------------------------------------
    # SAFE LYRICS EXTRACTION
    # Genius API sometimes returns:
    #   - Song object with .lyrics
    #   - dict with ["lyrics"]
    #   - None
    # --------------------------------------------------------
    lyrics_text = ""

    if song:
        try:
            if hasattr(song, "lyrics"):
                lyrics_text = song.lyrics
            elif isinstance(song, dict) and "lyrics" in song:
                lyrics_text = song["lyrics"]
        except:
            lyrics_text = ""

    # If Genius failed → YouTube transcript fallback
    if not lyrics_text.strip():
        print("✖ Genius failed → trying YouTube transcript…")
        transcript = get_youtube_transcript(yt_url)
        if transcript.strip():
            lyrics_text = transcript
            print("✔ FOUND (YouTube Transcript)")
        else:
            print("✖ No lyrics found anywhere.")

    meta.loc[i, "Lyrics"] = lyrics_text

    # Periodic checkpoint
    if i % 25 == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / max(elapsed, 1e-6)
        remaining = (len(meta) - i - 1) / max(rate, 1e-6)

        print(f"🎤 {i}/{len(meta)} | {rate:.2f} songs/sec | ETA {remaining/60:.1f} min")
        meta.to_csv(lyrics_checkpoint_path, index=False)
        print("💾 Saved lyrics checkpoint.")

    time.sleep(0.25)

# final save
meta.to_csv(lyrics_checkpoint_path, index=False)
print("\n🎉 FINISHED LYRICS COLLECTION")

# =====================================================================
# PART 2 — BATCHED EMBEDDINGS (FAST)
# =====================================================================
EMB_DIM = 1536
BATCH_SIZE = 32

if os.path.exists(embed_checkpoint_path):
    lyric_embeddings = np.load(embed_checkpoint_path)
    if lyric_embeddings.shape[0] != len(meta):
        lyric_embeddings = np.zeros((len(meta), EMB_DIM), dtype=np.float32)
else:
    lyric_embeddings = np.zeros((len(meta), EMB_DIM), dtype=np.float32)

texts = meta["Lyrics"].fillna("").astype(str).tolist()

indices_to_embed = [
    i for i in range(len(texts))
    if texts[i].strip() != "" and np.allclose(lyric_embeddings[i], 0)
]

print(f"\n🧠 Embedding {len(indices_to_embed)} songs…")

start_embed = time.time()

for start in range(0, len(indices_to_embed), BATCH_SIZE):
    batch_idx = indices_to_embed[start:start+BATCH_SIZE]
    batch_texts = [texts[i] for i in batch_idx]

    try:
        resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=batch_texts
        )
        vectors = [np.array(d.embedding, dtype=np.float32) for d in resp.data]

        for idx, vec in zip(batch_idx, vectors):
            lyric_embeddings[idx] = vec

    except Exception as e:
        print("Embedding batch error:", e)

    # checkpoint & progress
    if start % (BATCH_SIZE * 4) == 0:
        done = start + len(batch_idx)
        elapsed = time.time() - start_embed
        rate = done / max(elapsed, 1e-6)
        remaining = (len(indices_to_embed) - done) / max(rate, 1e-6)

        print(f"🧩 {done}/{len(indices_to_embed)} | {rate:.2f}/sec | ETA {remaining/60:.1f} min")
        np.save(embed_checkpoint_path, lyric_embeddings)
        print("💾 Saved embedding checkpoint.")

    time.sleep(0.1)

np.save(embed_checkpoint_path, lyric_embeddings)
print("\n🎉 Embedding complete!")

# =====================================================================
# PART 3 — FINAL MERGE
# =====================================================================
lyric_cols = [f"lyric_emb_{i}" for i in range(EMB_DIM)]
lyric_df = pd.DataFrame(lyric_embeddings, columns=lyric_cols)

final = pd.concat([meta.reset_index(drop=True), lyric_df], axis=1)
final.to_csv(output_path, index=False)

print("\n🎉 FINAL DATASET SAVED!")
print(output_path)
print("Final shape:", final.shape)
