import os
import logging
import numpy as np
import librosa
import joblib
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import audeer
import audonnx

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SAMPLING_RATE = 16000
CLASSIFIER_PATH = "test_classifier.pkl"
# Change this to ravdess path 
RAVDESS_ROOT = "path" 

# Emotion code to label
emotion_labels = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# Load ONNX model
url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
cache_root = audeer.mkdir('cache')
model_root = audeer.mkdir('model')
archive_path = audeer.download_url(url, cache_root, verbose=True)
audeer.extract_archive(archive_path, model_root)
model = audonnx.load(model_root)

# Feature extraction
def extract_features(filepath, model):
    signal, sr = librosa.load(filepath, sr=SAMPLING_RATE)
    if signal.ndim == 1:
        signal = np.expand_dims(signal, axis=0)
    output = model(signal, sampling_rate=sr)
    if 'hidden_states' in output and 'logits' in output:
        hidden = output['hidden_states'].flatten()
        logits = output['logits'].flatten()
        return np.concatenate([hidden, logits])
    else:
        logging.warning(f"Skipping {filepath} due to missing model outputs")
        return None

# Gather .wav files
wav_files = []
for root, _, files in os.walk(RAVDESS_ROOT):
    for file in files:
        if file.endswith(".wav"):
            wav_files.append(os.path.join(root, file))

# Extract features and labels
data = []
for idx, filepath in enumerate(wav_files):
    logging.info(f"Processing {idx+1}/{len(wav_files)}: {filepath}")
    parts = os.path.basename(filepath).split('-')
    try:
        emotion = emotion_labels[parts[2]]
        speaker = parts[6]
        features = extract_features(filepath, model)
        if features is not None:
            data.append((features, emotion, speaker))
    except (IndexError, KeyError):
        logging.warning(f"Skipping bad filename: {filepath}")

# Build arrays
features = np.vstack([d[0] for d in data])
emotions = [d[1] for d in data]
speakers = [d[2] for d in data]

# Train classifier
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
logo = LeaveOneGroupOut()
scores = cross_val_score(clf, features, emotions, groups=speakers, cv=logo)
logging.info(f"Cross-validation scores: {scores}")
logging.info(f"Mean accuracy: {np.mean(scores)}")

clf.fit(features, emotions)
joblib.dump(clf, CLASSIFIER_PATH)
logging.info(f"Classifier saved to {CLASSIFIER_PATH}")
