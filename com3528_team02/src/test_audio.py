import rospy
import numpy as np
from miro_msgs.msg import miroAudio

# Callback function to process audio and print amplitude
def audio_callback(msg):
    audio_data = np.array(msg.data, dtype=np.int16)  # Convert to NumPy array
    amplitude = np.max(np.abs(audio_data))  # Compute amplitude
    print(f"Amplitude: {amplitude}")  # Print the amplitude

# Initialize ROS node
rospy.init_node('miro_audio_amplitude', anonymous=True)
rospy.Subscriber('/miro/robot/audio', miroAudio, audio_callback)

print("Listening to MiRo's audio... Press Ctrl+C to stop.")
rospy.spin() 
