import audeer
import audonnx
import numpy as np
import librosa
import joblib
import os
import logging


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SAMPLING_RATE = 16000

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

# Classifier training or loading
script_dir = os.path.dirname(os.path.abspath(__file__))
classifier_path = os.path.join(script_dir, "ROS_Numpy_1.24.4.pkl")

def load_model():
    # Download and extract model
    url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
    cache_root = audeer.mkdir('cache')
    model_root = audeer.mkdir('model')
    archive_path = audeer.download_url(url, cache_root, verbose=True)
    audeer.extract_archive(archive_path, model_root)
    model = audonnx.load(model_root)

    return model

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

def run_model(wav_file, model):
    clf = joblib.load(classifier_path)
    logging.info("Loaded existing classifier.")

    features = extract_features(wav_file, model)
    predicted_emotion = clf.predict([features])
    logging.info(f"Predicted Emotion: {predicted_emotion[0]}")

    return predicted_emotion[0]