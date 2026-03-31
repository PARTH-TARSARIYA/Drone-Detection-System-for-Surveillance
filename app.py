import streamlit as st
import joblib
import numpy as np
import tempfile
import os
from moviepy.editor import VideoFileClip
from tensorflow.keras.models import load_model
from functions import extract_feature, extract_frame

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Drone Detection System", layout="wide")

st.markdown("""
    <h1 style='text-align: center;'>🚁 Drone Detection System</h1>
    <p style='text-align: center; font-size:18px;'>
    Upload a video to detect drone presence using Audio + Video AI
    </p>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    audio_model = joblib.load("audio_model.pkl")
    video_model = load_model("video_model.keras")
    audio_encoder = joblib.load('audio_encoder.pkl')
    video_encoder = joblib.load('video_encoder.pkl')
    return audio_model, video_model, audio_encoder, video_encoder

audio_model, video_model, audio_encoder, video_encoder = load_models()

# ---------------- SETTINGS ----------------
THRESHOLD = 0.30

# ---------------- FILE UPLOAD ----------------
uploaded_video = st.file_uploader("📤 Upload Video", type=["mp4", "avi", "mov"])

# ---------------- PROCESS ----------------
if uploaded_video is not None:

    # Create temp video file safely
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    video_path = temp_video.name
    temp_video.close()

    st.subheader("🎥 Uploaded Video")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.video(video_path)

    with st.spinner("🔍 Analyzing audio & video..."):

        video_clip = None

        try:
            # Load video
            video_clip = VideoFileClip(video_path)

            # ---------------- VIDEO ----------------
            video_frames = extract_frame(video_path)
            video_frames = np.expand_dims(video_frames, axis=0)

            video_probs = video_model.predict(video_frames, verbose=0)[0]
            video_class = np.argmax(video_probs)
            video_conf = float(np.max(video_probs))
            video_prediction = video_encoder.inverse_transform([video_class])[0]

            # ---------------- AUDIO ----------------
            audio_conf = None
            audio_prediction = None

            if video_clip.audio is not None:
                # Safe temp audio file
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                audio_path = temp_audio.name
                temp_audio.close()

                video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

                audio_features = extract_feature(audio_path)
                audio_features = np.array(audio_features).reshape(1, -1)

                audio_probs = audio_model.predict_proba(audio_features)[0]
                audio_class = np.argmax(audio_probs)
                audio_conf = float(np.max(audio_probs))
                audio_prediction = audio_encoder.inverse_transform([audio_class])[0]

                # delete audio file safely
                if os.path.exists(audio_path):
                    os.remove(audio_path)

                # ---------------- FUSION ----------------
                if video_prediction == audio_prediction:
                    final_result = video_prediction
                    final_confidence = (audio_conf + video_conf) / 2
                else:
                    if audio_conf >= video_conf:
                        final_result = audio_prediction
                        final_confidence = (0.7 * audio_conf) + (0.3 * video_conf)
                    else:
                        final_result = video_prediction
                        final_confidence = (0.7 * video_conf) + (0.3 * audio_conf)

            else:
                final_result = video_prediction
                final_confidence = video_conf

            # ---------------- THRESHOLD LOGIC ----------------
            if float(final_confidence) < THRESHOLD:
                final_result = "Threat" if final_result == "Non-Threat" else "Non-Threat"
                final_confidence = 1 - final_confidence

            final_confidence = final_confidence * 100

        except Exception as e:
            st.error(f"Error processing video: {e}")
            st.stop()

        finally:
            # 🔥 VERY IMPORTANT (fix WinError 32)
            if video_clip is not None:
                video_clip.close()

            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except:
                    pass

    # ---------------- RESULT UI ----------------
    st.subheader("📊 Detection Result")

    if final_result == "Threat":
        st.markdown(
            f"""
            <div style="
                font-size:36px;
                font-weight:bold;
                color:white;
                background-color:#ff4b4b;
                padding:20px;
                border-radius:12px;
                text-align:center;
                animation: blink 1s infinite;
            ">
            🚨 DRONE DETECTED<br>
            Confidence: {final_confidence:.2f}%
            </div>

            <style>
            @keyframes blink {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0; }}
                100% {{ opacity: 1; }}
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="
                font-size:32px;
                font-weight:bold;
                color:white;
                background-color:#28a745;
                padding:20px;
                border-radius:12px;
                text-align:center;
            ">
            ✅ NO DRONE DETECTED<br>
            Confidence: {final_confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    st.info("👆 Please upload a video file to start detection.")