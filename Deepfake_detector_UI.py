import streamlit as st
import numpy as np
import cv2
import face_recognition
import sounddevice as sd
from scipy.io.wavfile import write
import os
import mss
from d_voice import run_voice_analysis_streamlit
from d_image import MesoNet

# ========== Config ==========
REF_IMAGE = "gujju.jpg"
MESO_MODEL = "meso4.h5"
REF_AUDIO = "reference.wav"
TEST_AUDIO = "test_audio.wav"
SAMPLE_RATE = 16000
AUDIO_DURATION = 5
FRAME_SKIP = 5

# ========== Load Model ==========
meso = MesoNet(model_path=MESO_MODEL)

# ========== Load Reference Face ==========
try:
    ref_img = face_recognition.load_image_file(REF_IMAGE)
    ref_encodings = face_recognition.face_encodings(ref_img)
    if not ref_encodings:
        raise ValueError("No face found in reference image.")
    ref_enc = ref_encodings[0]
except Exception as e:
    st.error(f"❌ Error loading reference image: {e}")
    st.stop()

# ========== Audio Recorder ==========
def record_audio(filename, duration=AUDIO_DURATION):
    st.info(f"🎙️ Recording for {duration} seconds...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    write(filename, SAMPLE_RATE, audio)
    st.success(f"✅ Saved: {filename}")

# ========== Page Config ==========
st.set_page_config(page_title="Deepfake Detection Suite", layout="wide", page_icon="🧠")

# ========== Light / Dark Mode Toggle ==========
theme = st.sidebar.radio("🌗 Theme Mode", ["🌞 Light Mode", "🌙 Dark Mode"])

if theme == "🌙 Dark Mode":
    st.markdown(
        """
        <style>
            html, body, .main {
                background-color: #0E1117 !important;
                color: white !important;
            }
            .stButton>button {
                background-color: #31333F;
                color: white;
            }
            .stFileUploader {
                background-color: #1E1E1E;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        """
        <style>
            html, body, .main {
                background-color: white !important;
                color: black !important;
            }
            .stButton>button {
                background-color: #f0f0f0;
                color: black;
            }
            .stFileUploader {
                background-color: #ffffff;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ========== Page Title ==========
st.markdown("<h1 style='text-align: center;'>🧠 FAUXSHIELD</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

# ========== VOICE AUTHENTICATION ==========
st.subheader("🎙️ Voice Authentication & Deepfake Detection")
st.caption("Upload a reference voice or record live test voice to check authenticity.")

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        uploaded_ref = st.file_uploader("📤 Upload Reference Voice (WAV)", type=["wav"])
        if uploaded_ref:
            with open(REF_AUDIO, "wb") as f:
                f.write(uploaded_ref.read())
            st.audio(REF_AUDIO, format="audio/wav")
            st.success("✅ Reference voice uploaded successfully.")

    with col2:
        if st.button("🎙️ Record Test Voice"):
            record_audio(TEST_AUDIO)
        if os.path.exists(TEST_AUDIO):
            st.audio(TEST_AUDIO, format="audio/wav")

# ========== Run Analysis ==========
if os.path.exists(REF_AUDIO) and os.path.exists(TEST_AUDIO):
    if st.button("🚀 Run Voice Analysis"):
        with st.spinner("Analyzing voice sample..."):
            match, similarity, is_fake, zcr_score = run_voice_analysis_streamlit(REF_AUDIO, TEST_AUDIO)

        st.progress(min(int(similarity * 100), 100), text="🔍 Similarity Score")

        st.markdown(f"**🧠 Cosine Similarity:** `{similarity:.4f}`")
        st.markdown(f"**🎵 ZCR Score (Heuristic):** `{zcr_score:.4f}`")

        if match and not is_fake:
            st.success("✅ Real speaker matched. Voice is authentic.")
        elif not match and not is_fake:
           st.warning("⚠️ Unknown speaker, but sounds human.")
        elif match and is_fake:
            st.warning("⚠️ Voice matches speaker, but may be synthetically altered.")
        else:
            st.error("🚨 Likely deepfake or impersonation detected.")

st.markdown("<hr style='border:1px dashed gray'>", unsafe_allow_html=True)

# ========== IMAGE MONITORING ==========
st.subheader("🖥️ Screen-Based Face Monitoring & Deepfake Detection")
st.caption("Analyzes your live screen (e.g. Google Meet) to detect deepfake faces.")

# Persistent state
if "detection_running" not in st.session_state:
    st.session_state.detection_running = False

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Face Detection"):
        st.session_state.detection_running = True
with col2:
    if st.button("⏹️ Stop Face Detection"):
        st.session_state.detection_running = False

placeholder = st.empty()

if st.session_state.detection_running:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        frame_count = 0

        while st.session_state.detection_running:
            frame = np.array(sct.grab(monitor))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

            if frame_count % FRAME_SKIP == 0:
                face_locs = face_recognition.face_locations(rgb_frame)
                face_encs = face_recognition.face_encodings(rgb_frame, face_locs)

                for (top, right, bottom, left), enc in zip(face_locs, face_encs):
                    matched = face_recognition.compare_faces([ref_enc], enc, tolerance=0.6)[0]
                    face_crop = rgb_frame[top:bottom, left:right]

                    if matched:
                        prob = meso.predict(face_crop)
                        label = "REAL" if prob > 0.5 else "FAKE"
                        result = f"✅ Match – {label} ({prob*100:.1f}%)"
                        color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
                    else:
                        result = "❌ Unknown Face"
                        color = (255, 0, 0)

                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, result, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    st.info(result)

            placeholder.image(frame, channels="BGR", use_container_width=True)
            frame_count += 1

# ========== Footer ==========
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("🛡️ Made with 💡 by YourName | Powered by SpeechBrain, FaceNet, and MesoNet.")
