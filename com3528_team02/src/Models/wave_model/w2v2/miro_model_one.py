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
#classifier_path = os.path.join(script_dir, "ROS_Numpy_1.24.4.pkl")
current_os = os.getcwd()
classifier_path = f"{current_os}/test_classifier.pkl"
# classifier_path = "~/mdk/catkin_ws/src/COM3528_2025_Team02/com3528_team02/src/test_classifier.pkl"


import pickle

# with open(classifier_path, "rb") as f:
#     data = joblib.load(f)

# with open(classifier_path, "rb") as f:
#     print(f.read(10))
#     data = pickle.load(f)



def load_model():
    model_onnx_path = f"{os.getcwd()}/model/"
    model = audonnx.load(model_onnx_path)
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
    print(f"here ------- {classifier_path}")
    clf = joblib.load(classifier_path)
    print("d")
    logging.info("Loaded existing classifier.")

    features = extract_features(wav_file, model)
    predicted_emotion = clf.predict([features])
    logging.info(f"Predicted Emotion: {predicted_emotion[0]}")

    return predicted_emotion[0]