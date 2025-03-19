import numpy as np
from com3528.src.sound_localizer import SoundLocalizer

def main():
    
    """ 
    Simulation of input microphone data (this would come from Miro's microphone system)
    Should later be replaced with actual microphone data 
    """
    left_ear_data = np.random.randn(16000)
    right_ear_data = np.random.randn(16000)
    tail_data = np.random.randn(16000)

    # Initialize the SoundLocaliser
    sound_localiser = SoundLocalizer(mic_distance=0.1, sample_rate=16000)

    # Process the data from the microphones
    direction_x, direction_y = sound_localiser.process_data(left_ear_data, right_ear_data, tail_data)

    if direction_x is not None and direction_y is not None:
        print(f"Moving robot towards sound source at coordinates: ({direction_x}, {direction_y})")
    else:
        print("No human speech detected.")

if __name__ == "__main__":
    main()
