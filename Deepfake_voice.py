# deepfake_voice.py

import numpy as np
import soundfile as sf
import torch
from scipy.spatial.distance import cosine
from speechbrain.inference import SpeakerRecognition
from pydub import AudioSegment

# ========== Config ==========
SIMILARITY_THRESHOLD = 0.65  # Adjust as needed
ZCR_THRESHOLD = 0.40        # Adjust if needed
SAMPLE_RATE = 16000

# ========== Load Models ==========
spkrec = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec"
)

# ========== Embedding Extraction ==========
def get_embedding(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
        samples = torch.tensor(audio.get_array_of_samples()).float()
        samples = samples / 32768.0
        signal = samples.unsqueeze(0)

        if signal.shape[1] < SAMPLE_RATE:
            print("⚠️ Audio too short for reliable embedding.")
            return None

        signal = signal.to(spkrec.device)
        emb = spkrec.encode_batch(signal).squeeze()
        return emb.detach().cpu().numpy().flatten()
    except Exception as e:
        print(f"❌ Failed to get embedding for {audio_path}: {e}")
        return None

# ========== Compare Embeddings ==========
def compare_embeddings(emb1, emb2, threshold=SIMILARITY_THRESHOLD):
    if emb1 is None or emb2 is None:
        return False, 0.0
    sim = 1 - cosine(emb1, emb2)
    return sim >= threshold, sim

# ========== Simple Deepfake Detection (ZCR Heuristic) ==========
def detect_deepfake_audio(audio_path):
    try:
        y, sr = sf.read(audio_path)
        zcr = np.mean(np.abs(np.diff(np.sign(y))))
        print(f"🧪 ZCR (spoof heuristic): {zcr:.4f}")
        return zcr > ZCR_THRESHOLD
    except Exception as e:
        print(f"❌ Error during ZCR: {e}")
        return True

# ========== Combined Function for Streamlit ==========
def run_voice_analysis_streamlit(reference_path, test_path):
    ref_emb = get_embedding(reference_path)
    test_emb = get_embedding(test_path)

    if ref_emb is None or test_emb is None:
        return False, 0.0, True, 0.0  # default values

    match, similarity = compare_embeddings(ref_emb, test_emb)
    is_fake = detect_deepfake_audio(test_path)

    try:
        y, _ = sf.read(test_path)
        zcr_score = np.mean(np.abs(np.diff(np.sign(y))))
    except Exception as e:
        print(f"Error calculating ZCR: {e}")
        zcr_score = 0.0

    return match, similarity, is_fake, zcr_score
