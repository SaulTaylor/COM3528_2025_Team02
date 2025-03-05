#!/usr/bin/env python3

import numpy as np
import rospy
from std_msgs.msg import Int16MultiArray

class SoundLocalizer:
    def __init__(self, mic_distance=0.1):
        self.mic_distance = mic_distance
        self.speed_of_sound = 343  # Speed of sound in m/s
        self.no_of_mics = 4
        self.mic_data = np.zeros((self.no_of_mics, 500))  # Assuming 500 samples per update

    def callback_mics(self, data):
        # Update mic data directly from ROS callback
        new_data = np.array(data.data).reshape((self.no_of_mics, -1))
        self.mic_data = np.hstack((self.mic_data[:, -500:], new_data))

    def gcc(self, mic1, mic2):
        # Generalized Cross-Correlation implemented as in AudioEngine.py
        pad1 = np.zeros(len(mic1))
        pad2 = np.zeros(len(mic2))
        s1 = np.hstack([mic1, pad1])
        s2 = np.hstack([pad2, mic2])
        f_s1 = np.fft.fft(s1)
        f_s2 = np.fft.fft(s2)
        f_s2c = np.conj(f_s2)
        f_s = f_s1 * f_s2c
        denom = np.abs(f_s)
        denom[denom == 0] = 1e-10
        f_s /= denom
        correlation = np.fft.ifft(f_s)
        delay = np.argmax(np.abs(correlation)) - len(mic1)
        return delay

    def process_data(self):
        # Apply GCC to updated mic data
        delay_left_right = self.gcc(self.mic_data[0], self.mic_data[1])
        delay_left_tail = self.gcc(self.mic_data[0], self.mic_data[3])
        delay_right_tail = self.gcc(self.mic_data[1], self.mic_data[3])

        # Convert delays to angles using small angle approximation
        angle_left_right = (delay_left_right / self.speed_of_sound) * self.mic_distance
        angle_left_tail = (delay_left_tail / self.speed_of_sound) * self.mic_distance
        angle_right_tail = (delay_right_tail / self.speed_of_sound) * self.mic_distance

        # Simple average of angles as a naive triangulation approach
        estimated_direction = np.mean([angle_left_right, angle_left_tail, angle_right_tail])
        print("Estimated direction of sound:", estimated_direction)
        return estimated_direction

if __name__ == '__main__':
    rospy.init_node('sound_localizer')
    localizer = SoundLocalizer()
    rospy.Subscriber("/miro/sensors/mics", Int16MultiArray, localizer.callback_mics, queue_size=1, tcp_nodelay=True)
    rospy.spin()  # Keep node alive and listening for data
