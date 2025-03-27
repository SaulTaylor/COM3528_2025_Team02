#!/usr/bin/env python3

import numpy as np
import os
import time
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import rospy
import miro2 as miro
from AudioEngine import DetectAudioEngine
from std_msgs.msg import Int16MultiArray
from geometry_msgs.msg import Twist, TwistStamped
from scipy.signal import find_peaks
import pandas as pd
from scipy.io import wavfile
import tf.transformations as tf_trans  # Converts quaternion to Euler angles
from nav_msgs.msg import Odometry



""" ----------------------------- FILE OVERVIEW -----------------------------
This file aims to detect an audio event, run it through a model to get the emotion,
if that emotion is 'angry', 'sad' or 'fearful' then locate where the human sound is
coming from and move towards them. 

The code currently performs the following main tasks:

1. Subscribes to the robot's microphone data stream and buffers incoming audio.
2. Identifies peaks in the audio signals to detect sound events.
3. Uses cross-correlation and generalized cross-correlation to estimate time 
   delays between microphones and infer the direction (angle) of the sound source.
4. Turns the robot toward the detected sound and moves it forward as long as the
   sound remains above a minimum threshold.
5. Records a segment of audio as a .wav file when sound is detected.
6. (Planned) Runs the recorded audio through a machine learning model (e.g. Whisper
   or sound classification model) to classify or transcribe the sound and get a string label.
7. (Planned) The robot can then react based on the label returned
   (e.g. approach a person who is arguing).
The class `SoundLocalizer` handles all functionality including subscribing to
ROS topics, audio processing, angle estimation, robot movement, and model inference.
"""

class SoundLocalizer:
    def __init__(self, mic_distance=0.1):

        self.mic_distance = mic_distance

        # sets up mic buffer
        self.x_len = 40000
        self.no_of_mics = 4
        self.input_mics = np.zeros((self.x_len, self.no_of_mics))

        # which miro
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

        # subscribers
        self.sub_mics = rospy.Subscriber(topic_base_name + "/sensors/mics",
                                         Int16MultiArray, self.callback_mics, queue_size=1, tcp_nodelay=True)

        # publishers
        self.pub_push = rospy.Publisher(topic_base_name + "/core/mpg/push", miro.msg.push, queue_size=0)
        self.pub_wheels = rospy.Publisher(topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0)

        # prepare push message
        self.msg_push = miro.msg.push()
        self.msg_push.link = miro.constants.LINK_HEAD
        self.msg_push.flags = (miro.constants.PUSH_FLAG_NO_TRANSLATION + miro.constants.PUSH_FLAG_VELOCITY)

        # time
        self.msg_wheels = TwistStamped()

        self.left_ear_data = np.flipud(self.input_mics[:, 0])
        self.right_ear_data = np.flipud(self.input_mics[:, 1])
        self.head_data = np.flipud(self.input_mics[:, 2])
        self.tail_data = np.flipud(self.input_mics[:, 3])

        # flags for averaging and rotating
        self.averaging = False
        self.rotating = False

        # Running average stuff
        self.t1_values = []
        self.t2_values = []

        self.current_yaw = 0.0
        rospy.Subscriber(topic_base_name + "/sensors/odom", Odometry, self.callback_pose)


        print("init success")

    def callback_pose(self, msg):
        orientation_q = msg.pose.pose.orientation
        q = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = tf_trans.euler_from_quaternion(q)  # Get yaw in radians
        self.current_yaw = yaw

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

    @staticmethod
    def block_data(data, block_size=500):
        # Calculate the number of blocks
        num_of_blocks = len(data) // block_size

        blocks = []

        for i in range(num_of_blocks):
            start = i * block_size
            end = start + block_size

            block = data[start:end]

            # Add the block to the list of blocks
            blocks.append(block)

        return np.array(blocks)

    @staticmethod
    def find_high_peaks(audio_data):
        # Height parameter acts as a threshold for which peaks are detected
        # The higher the less sensitive to background noise
        peaks, _ = find_peaks(audio_data, height=0.7)

        if peaks == []:
            print("No peaks found above height 0.7 in audio input")
            return None
        else:
            return peaks

    @staticmethod
    def create_block(index, data, block_size=500):
        # take the data around an index and create a block half of block size before and after the index
        block = data[index - block_size // 2:index + block_size // 2]

        return block

    def process_data(self):

        # get the high points
        peak_l = self.find_high_peaks(self.left_ear_data)
        peak_r = self.find_high_peaks(self.right_ear_data)
        peak_t = self.find_high_peaks(self.tail_data)

        # find a common points
        # Convert to sets
        set_l_peak = set(peak_l)
        set_r_peak = set(peak_r)
        set_t_peak = set(peak_t)

        # Try to find common high points and convert to blocks
        try:
            common_high_points = set_l_peak.intersection(set_r_peak, set_t_peak)

            common_values_l = [self.left_ear_data[point] for point in common_high_points]
            common_values_r = [self.right_ear_data[point] for point in common_high_points]
            common_values_t = [self.tail_data[point] for point in common_high_points]

            # Calculate the sum of values for each common high point
            # By doing this we can find the common high point with the largest accumulative value
            sum_values = [self.left_ear_data[point] + self.right_ear_data[point] + self.tail_data[point] for point in
                          common_high_points]

            # Find the index of the maximum sum
            max_index = np.argmax(sum_values)

            # Get the common high point with the largest accumulative value
            max_common_high_point = list(common_high_points)[max_index]

            # Threshold acts as a second filter to height parameter in find_high_peaks
            # Works for the common high points rather than the regular high points
            threshold = 700
            # check that common values reach threshold
            if max(common_values_l) < threshold or max(common_values_r) < threshold or max(common_values_t) < threshold:

                print("Common points exceed threshold and audio file is saved as .wav")
                self.save_audio_to_wav(self.left_ear_data)

                return None

            # Get block around max common high point
            max_common_block_l = self.create_block(max_common_high_point, self.left_ear_data)
            max_common_block_r = self.create_block(max_common_high_point, self.right_ear_data)
            max_common_block_t = self.create_block(max_common_high_point, self.tail_data)

            x1_l_r = np.correlate(max_common_block_l, max_common_block_r, mode='same')
            x2_l_t  = np.correlate(max_common_block_l, max_common_block_t, mode='same')
            x_r_t  = np.correlate(max_common_block_r, max_common_block_t, mode='same')

            r1_hat = np.argmax(x1_l_r) 
            r2_hat = np.argmax(x2_l_t)

            t1_1 = np.cos(r1_hat * 343) / .1
            t2_1 = np.cos(r2_hat * 343) / .25

            print(t1_1, t2_1)

            return t1_1, t2_1
        except Exception as e:
            print("No common high points")
            return None, None

    def callback_mics(self, data):
    # data for angular calculation
        # data for display
        data = np.asarray(data.data)
        # 500 samples from each mics
        data = np.transpose(data.reshape((self.no_of_mics, 500)))  # after this step each row is a sample and each
        # column is the mag. at that sample time for each mic
        data = np.flipud(data)  # flips as the data comes in reverse order
        self.input_mics = np.vstack((data, self.input_mics[:self.x_len - 500, :]))
        self.left_ear_data = np.flipud(self.input_mics[:, 0])
        self.right_ear_data = np.flipud(self.input_mics[:, 1])
        self.head_data = np.flipud(self.input_mics[:, 2])
        self.tail_data = np.flipud(self.input_mics[:, 3])

        global av1, av2

        if not self.rotating:
            t1, t2, = None, None
            try:
                t1, t2 = self.process_data()
            except Exception as e:
                t1, t2 = None, None

            # running average for t1 and t2 so long as there are high points
            # being found then we will assume their from the same source
            # this should also reduce the error as a result of noise

            # if there's a value  and  we are averaging start tracking
            # if not t1 is None and not self.averaging:
            #     self.t1_values.append(t1)
            #     self.t2_values.append(t2)

            if t1 is not None:
                self.t1_values.append(t1)
                self.t2_values.append(t2)

            # if there's no value and we are averaging then stop tracking
            # as there is no sound source (no high point found)
            if len(self.t1_values) > 0:
                try:
                    # average the values using running average lists
                    av1, av2 = np.average(self.t1_values), np.average(self.t2_values)
                    print('running average for t1, t2: ', av1, av2)
                except Exception as e:
                    print(e)

                print('turning then moving toward sound')
                an = self.estimate_angle(av1, av2)
                self.move_to_sound(an)
                self.t1_values = []
                self.t2_values = []

            self.averaging = False
            #     an = self.estimate_angle(av1, av2)
            #     self.move_to_sound(an)
            #     time.sleep(2)
            #     self.t1_values = []
            #     self.t2_values = []

            # # sets averaging to true if none and not already averaging
            # self.averaging = t1 is None and not self.averaging


    def save_audio_to_wav(self, filename="get_emotion_from_audio.wav", mic_index=0, sample_rate=16000):
        """
        Save audio from left ear (index 0) microphone to a .wav file
        """
        audio_data = self.input_mics[:, mic_index]
        audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)  # Normalize to int16
        wavfile.write(filename, sample_rate, audio_data)
        print(f"Saved audio to {filename}")


    @staticmethod
    def estimate_angle(t1, t2):

        # t1 time delay between left and right ear
          # positive then sound source is on right ear
          # negative then sound source is on left ear

        # t2 time delay between left ear and tail
          # positive if in front
          # negative if behind 

        angle = 0
        if t2 >= 0:  # then the sound is coming from behind
            angle += 0
        if t2 < 0:
            angle += 180 
        if t1 > 0:
            angle += 45 * abs(t1)/2
        if t1 < 0:
            angle -= 45 * abs(t2)/2
        
        print("angle (degrees) to sound source: ", angle)
        if angle > 360:
            angle = angle - 360
        
        return np.deg2rad(angle)

    def move_to_sound(self, azimuth, min_intensity=500):
        """
        Moves MiRo toward the sound source, continuously checking if it remains on the same path.
        
        Parameters:
        - azimuth: Angle to the sound source.
        - min_intensity: Minimum sound threshold to indicate MiRo is near the human.
        """
        self.rotating = True

        # Turn to the sound source
        target_degrees = np.rad2deg(azimuth)  # Convert radians to degrees for easier debugging
        print(f"Turning to azimuth {np.rad2deg(azimuth)} degrees")
        angular_speed = 0.6
        rotation_time = abs(target_degrees) / (angular_speed * 57.3)  # Time required to turn
        print(f"I'm rotating with speed {angular_speed} for {rotation_time} seconds")

        # start_time = rospy.Time.now().to_sec()
        # while (rospy.Time.now().to_sec() - start_time) < rotation_time:
        #     self.msg_wheels.twist.linear.x = 0.0  # No forward movement
        #     self.msg_wheels.twist.angular.z = np.sign(azimuth) * angular_speed  # Rotate in the correct direction
        #     print("self.msg_wheels.twist.angular.z: ", self.msg_wheels.twist.angular.z)
        #     self.pub_wheels.publish(self.msg_wheels)
        #     rospy.sleep(0.1)  # Keep checking
        # Get angle before turning

        start_yaw = self.current_yaw

        start_time = rospy.Time.now().to_sec()
        print(f"I'm about to turn for {rotation_time} seconds")
        while (rospy.Time.now().to_sec() - start_time) < rotation_time:
            self.msg_wheels.twist.linear.x = 0.0
            self.msg_wheels.twist.angular.z = np.sign(azimuth) * angular_speed
            self.pub_wheels.publish(self.msg_wheels)
            print(f"I have been rotating for : {rospy.Time.now().to_sec() - start_time} seconds")
            rospy.sleep(0.1)

        # Get angle after turning
        end_yaw = self.current_yaw

        # Calculate change in angle
        delta_yaw = end_yaw - start_yaw
        delta_degrees = (np.rad2deg(delta_yaw) + 360) % 360  # Normalize between 0-360
        if delta_degrees > 180:
            delta_degrees -= 360  # Normalize to -180 to 180

        print(f"Actual angle turned: {delta_degrees:.2f} degrees")

        # Stop rotation
        self.msg_wheels.twist.angular.z = 0.0
        self.pub_wheels.publish(self.msg_wheels)
        print(f"I think I have turned {target_degrees} degrees")

        print(f"I think i have turned {np.rad2deg(azimuth)} degrees")
        self.rotating = False # finished rotating

        # Move forward while checking sound intensity
        print("Moving toward the sound source")
        while True: 
            # check the latest sound intensity
            sound_intensity = np.max([
                np.max(self.left_ear_data), 
                np.max(self.right_ear_data), 
                np.max(self.tail_data)
            ])

            if sound_intensity >= min_intensity:
                print("Tracking active sound source...")
                self.msg_wheels.header.stamp = rospy.Time.now()
                self.msg_wheels.twist.linear.x = 0.1  # Move forward
                self.msg_wheels.twist.angular.z = 0.0  # No extra rotation yet
                self.pub_wheels.publish(self.msg_wheels)
            else: 
                print("Sound weakend or stopped, pausing movement")
                print("sound_intensity: ", sound_intensity )
                break

            # Move forward in the estimated direction
            self.msg_wheels.twist.linear.x = 0.1
            self.msg_wheels.twist.angular.z = 0.0
            self.pub_wheels.publish(self.msg_wheels)
            rospy.sleep(0.1)  # Short delay before rechecking

        # Stop moving
        self.msg_wheels.twist.linear.x = 0.0    
        self.pub_wheels.publish(self.msg_wheels)



# Example of using the class
if __name__ == '__main__':
    print("Initialising")
    rospy.init_node('sound_localizer')
    AudioEng = DetectAudioEngine()
    localizer = SoundLocalizer()
    direction = localizer.process_data()

    rospy.spin()  # Keeps Python from exiting until this node is stopped
