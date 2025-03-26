import audeer
import audonnx
import numpy as np
import pyaudio
import wave
import librosa
import audb

# Download and extract model
url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
cache_root = audeer.mkdir('cache')
model_root = audeer.mkdir('model')

archive_path = audeer.download_url(url, cache_root, verbose=True)
audeer.extract_archive(archive_path, model_root)
model = audonnx.load(model_root)

# Recording user audio
def recordAudio(seconds=5, sampling_rate=16000):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    filename = "output.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=sampling_rate,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for the specified duration
    for i in range(0, int(sampling_rate / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(sampling_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    return filename

# Function to process recorded audio with the model
def processAudioWithModel():
    audio_filename = recordAudio()
    signal, sr = librosa.load(audio_filename, sr=None)  # Load the audio file
    result = model(signal, sr)
    vad_scores = result['logits'][0]  # Extract VAD scores from logits assuming they are stored here
    print(result)


# Example of how to use the recording and model processing
processAudioWithModel()
