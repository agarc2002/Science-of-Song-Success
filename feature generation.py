import os
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

# === CONFIG ===
AUDIO_DIR = r"C:\Users\rohan\Downloads\STA 160\audio_data"
OUTPUT_DIR = r"C:\Users\rohan\Downloads\STA 160\mert_outputs"
BATCH_SIZE = 100
SAMPLING_RATE = 24000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load MERT model ===
print("🎶 Loading MERT model...")
processor = AutoProcessor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("✅ Model loaded on:", device)

# === Helper ===
def extract_embedding(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLING_RATE)
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return emb
    except Exception as e:
        print(f"❌ Failed: {file_path} ({e})")
        return None

# === Load progress if exists ===
completed_files = set()
progress_file = os.path.join(OUTPUT_DIR, "completed_files.txt")
if os.path.exists(progress_file):
    with open(progress_file, "r") as f:
        completed_files = set(f.read().splitlines())

all_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith((".opus", ".mp3", ".wav", ".m4a"))]
pending_files = [f for f in all_files if f not in completed_files]
print(f"🎧 {len(pending_files)} files pending out of {len(all_files)} total.")

# === Process in batches ===
for i in range(0, len(pending_files), BATCH_SIZE):
    batch = pending_files[i:i + BATCH_SIZE]
    embeddings, meta = [], []

    print(f"\n🚀 Processing batch {i//BATCH_SIZE + 1} / {(len(pending_files)//BATCH_SIZE)+1} ({len(batch)} files)")
    for file in tqdm(batch):
        path = os.path.join(AUDIO_DIR, file)
        emb = extract_embedding(path)
        if emb is not None:
            embeddings.append(emb)
            meta.append({"filename": file})
            completed_files.add(file)
            with open(progress_file, "a") as f:
                f.write(file + "\n")

    if embeddings:
        np.save(os.path.join(OUTPUT_DIR, f"batch_{i//BATCH_SIZE+1}_embeddings.npy"), np.array(embeddings))
        pd.DataFrame(meta).to_csv(os.path.join(OUTPUT_DIR, f"batch_{i//BATCH_SIZE+1}_metadata.csv"), index=False)
        print(f"✅ Saved batch {i//BATCH_SIZE+1} with {len(embeddings)} embeddings.")

print("🎉 All batches processed successfully!")
