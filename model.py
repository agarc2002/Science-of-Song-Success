import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from tqdm import tqdm

# === 1. Paths ===
AUDIO_DIR = r"C:\Users\rohan\Downloads\STA 160\audio_data"   # folder containing .opus/.mp3 files
OUTPUT_EMBEDDINGS = "yamnet_embeddings.npy"
OUTPUT_METADATA = "yamnet_metadata.csv"

# === 2. Load YAMNet from TensorFlow Hub ===
print("🔄 Loading YAMNet model...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("✅ YAMNet loaded successfully!")

# === 3. Utility: Extract embeddings from one file ===
def extract_yamnet_embedding(file_path):
    try:
        wav_data, sr = librosa.load(file_path, sr=16000)  # resample to 16kHz
        scores, embeddings, spectrogram = yamnet_model(wav_data)
        # average across time frames to get a single embedding vector
        return np.mean(embeddings, axis=0)
    except Exception as e:
        print(f"❌ Failed to process {file_path}: {e}")
        return None

# === 4. Process all files ===
embeddings = []
metadata = []

print(f"🎵 Processing audio files from {AUDIO_DIR}...")
for file in tqdm(os.listdir(AUDIO_DIR)):
    if file.endswith((".opus", ".mp3", ".wav", ".m4a")):
        path = os.path.join(AUDIO_DIR, file)
        emb = extract_yamnet_embedding(path)
        if emb is not None:
            embeddings.append(emb)
            metadata.append({"filename": file})

# === 5. Save results ===
if len(embeddings) > 0:
    np.save(OUTPUT_EMBEDDINGS, np.array(embeddings))
    pd.DataFrame(metadata).to_csv(OUTPUT_METADATA, index=False)
    print(f"✅ Saved {len(embeddings)} embeddings to {OUTPUT_EMBEDDINGS}")
    print(f"✅ Metadata saved to {OUTPUT_METADATA}")
else:
    print("⚠️ No embeddings extracted — check audio directory path.")
