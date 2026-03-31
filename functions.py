# Audio Processing.
import cv2
import librosa
import numpy as np

def extract_feature(file_path):
    signal, sr = librosa.load(file_path, sr = 22100)  # sr means sample rating, it converts audio into numerical amplitudes.
    
    mfcc = librosa.feature.mfcc(y = signal, sr = sr, n_mfcc = 13)
    mfcc = np.mean(mfcc.T, axis = 0)

    return mfcc


# Extract Frame from image.

def extract_frame(video_path, target_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // target_frames)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if count % step == 0:
            frame = cv2.resize(frame, (64, 64))
            frame = frame.astype('float32') / 255.0
            frames.append(frame)

        count += 1

    cap.release()

    if len(frames) >= target_frames:
        frames = frames[:target_frames]
    else:
        padding = [np.zeros((64, 64, 3), dtype='float32')] * (target_frames - len(frames))
        frames.extend(padding)

    return np.array(frames)