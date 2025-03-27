import audeer
import audonnx
import numpy as np
import pyaudio
import wave
import librosa
import audb
import pandas as pd
import joblib
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os
import logging


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SAMPLING_RATE = 16000

# Download and extract model
url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
cache_root = audeer.mkdir('cache')
model_root = audeer.mkdir('model')
archive_path = audeer.download_url(url, cache_root, verbose=True)
audeer.extract_archive(archive_path, model_root)
model = audonnx.load(model_root)

# Mapping of RAVDESS emotion codes to descriptions
emotion_labels = {
    '01': 'neutral', 
    '02': 'calm', 
    '03': 'happy', 
    '04': 'sad',
    '05': 'angry', 
    '06': 'fearful', 
    '07': 'disgust', 
    '08': 'surprised'
}

# Function to record audio
def recordAudio(seconds=5, sampling_rate=SAMPLING_RATE):
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    filename = "output.wav"
    p = pyaudio.PyAudio()
    logging.info('Recording')
    stream = p.open(format=sample_format, channels=channels, rate=sampling_rate, frames_per_buffer=chunk, input=True)
    frames = []
    for _ in range(int(sampling_rate / chunk * seconds)):
        frames.append(stream.read(chunk))
    stream.stop_stream()
    stream.close()
    p.terminate()
    logging.info('Finished recording')
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(sampling_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

# Extract emotion and speaker from filenames
def extract_labels_and_features(db, model):
    data = []
    for index, file in enumerate(db.files):
        logging.info(f"Processing file {index + 1}/{len(db.files)}: {file}")
        filepath = os.path.join(db.root, file)
        features = extract_features(filepath, model)
        parts = file.split('-')
        emotion = emotion_labels[parts[2]]
        speaker = parts[6]
        data.append((features, emotion, speaker))
    return pd.DataFrame(data, columns=['features', 'emotion', 'speaker'])

def extract_features(filepath, model):
    logging.info(f"Extracting features from {filepath}")
    signal, sr = librosa.load(filepath, sr=SAMPLING_RATE)
    if signal.ndim == 1:
        signal = np.expand_dims(signal, axis=0)
    output = model(signal, sampling_rate=sr)
    
    if 'hidden_states' in output and 'logits' in output:
        hidden_states = output['hidden_states']
        vad_scores = output['logits']  # Assuming logits are VAD scores
        if hidden_states.ndim > 1:
            hidden_states = hidden_states.flatten()  # Flatten the array
        if vad_scores.ndim > 1:
            vad_scores = vad_scores.flatten()  # Flatten the array
        
        # Concatenate flattened hidden states and VAD scores
        features = np.concatenate([hidden_states, vad_scores])
        print(features)
        return features
    else:
        logging.error(f"Expected model outputs not found in {filepath}")
        return None
    
# Function to predict emotion from recorded audio
def predict_emotion(model):
    audio_filename = recordAudio()
    features = extract_features(audio_filename, model)
    predicted_emotion = clf.predict([features])
    logging.info(f"Predicted Emotion: {predicted_emotion[0]}")
    
 # Classifier training or loading
classifier_path = "ROS_Numpy_1.24.4.pkl"
if not os.path.exists(classifier_path):
    # Load RAVDESS dataset
    # Path to your extracted RAVDESS folder
    ravdess_root = "/home/student/mdk/catkin_ws/src/COM3528_2025_Team02/com3528/Models/SER Model/Real-Time-Speech-Emotion-Recognition/Dataset/speech-emotion-recognition-ravdess-data"  # <-- Update this path as needed

    # Gather all wav files
    wav_files = []
    for root, _, files in os.walk(ravdess_root):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))

    # Extract data
    data = []
    for index, filepath in enumerate(wav_files):
        logging.info(f"Processing file {index + 1}/{len(wav_files)}: {filepath}")
        features = extract_features(filepath, model)
        parts = os.path.basename(filepath).split('-')
        try:
            emotion = emotion_labels[parts[2]]
            speaker = parts[6]
            data.append((features, emotion, speaker))
        except (IndexError, KeyError):
            logging.warning(f"Skipping unrecognized file format: {filepath}")

    # Extract data
    features = np.vstack([d[0] for d in data if isinstance(d[0], np.ndarray)])
    emotions = [d[1] for d in data if isinstance(d[0], np.ndarray)]
    speakers = [d[2] for d in data if isinstance(d[0], np.ndarray)]


    # Log the type and shape of features to ensure they are correct
    logging.info(f"Final features shape: {features.shape}")
    logging.info(f"Features type: {features.dtype}")

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    logo = LeaveOneGroupOut()
    scores = cross_val_score(clf, features, emotions, groups=speakers, cv=logo)
    logging.info(f"Cross-validation scores: {scores}")
    logging.info(f"Mean accuracy: {np.mean(scores)}")
    clf.fit(features, emotions)
    joblib.dump(clf, classifier_path)
    logging.info("Classifier trained and saved.")

else:
    clf = joblib.load(classifier_path)
    logging.info("Loaded existing classifier.")
    predict_emotion(model)

# def run_model(wav_file):
#    clf = joblib.load(classifier_path)
#    logging.info("Loaded existing classifier.")
#    predict_emotion_miro(model, wav_file)

# def predict_emotion_miro(model, wav_file):
#    features = extract_features(wav_file, model)
#    predicted_emotion = clf.predict([features])
#    logging.info(f"Predicted Emotion: {predicted_emotion[0]}")

#    return predicted_emotion[0]