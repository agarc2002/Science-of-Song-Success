import os
import numpy as np
import librosa

def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration_ms = len(y) / sr * 1000

        # === Tempo ===
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # === RMS Energy / Loudness ===
        rms = librosa.feature.rms(y=y)[0]
        energy = np.mean(rms)
        loudness_db = 10 * np.log10(np.mean(rms ** 2) + 1e-10)

        # === Spectral features ===
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # === Acousticness ===
        acousticness = 1 - np.tanh(spec_centroid / 5000)

        # === Danceability ===
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        tempo_std = np.std(np.mean(tempogram, axis=0))
        danceability = np.clip(1 - tempo_std / 10, 0, 1)

        # === Valence ===
        valence = np.clip((tempo / 200) * (spec_centroid / 4000), 0, 1)

        # === Speechiness ===
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        speechiness = np.clip(0.5 * spectral_flatness + 0.5 * zcr, 0, 1)

        # === Instrumentalness ===
        instrumentalness = np.clip(acousticness * (1 - speechiness), 0, 1)

        # === Key & Mode ===
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key = int(np.argmax(chroma_mean))
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        mode = int(np.mean(tonnetz) > 0)

        # === Time Signature ===
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        autocorr = librosa.autocorrelate(onset_env)
        peaks = np.diff(np.sign(np.diff(autocorr))) < 0
        bar_peaks = np.sum(peaks)
        time_signature = 4 if bar_peaks < 10 else 3

        return {
            "filename": os.path.basename(file_path),
            "duration_ms": duration_ms,
            "tempo": tempo,
            "loudness": loudness_db,
            "energy": energy,
            "danceability": danceability,
            "acousticness": acousticness,
            "speechiness": speechiness,
            "instrumentalness": instrumentalness,
            "valence": valence,
            "key": key,
            "mode": mode,
            "time_signature": time_signature,
            "spectral_contrast": spec_contrast,
            "zcr": zcr,
            "spec_centroid": spec_centroid,
            "spec_bandwidth": spec_bandwidth,
        }

    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None