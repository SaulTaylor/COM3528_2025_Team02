import numpy as np
import webrtcvad
import time
import os
import numpy as np
import rospy
import miro2 as miro
import geometry_msgs
from AudioEngine import DetectAudioEngine
from std_msgs.msg import Int16MultiArray
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from geometry_msgs.msg import Twist, TwistStamped
import time

class SoundLocalizer:
    def __init__(self, mic_distance=0.1, sample_rate=16000, frame_duration_ms=30):
        self.mic_distance = mic_distance
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms  # Frame size in milliseconds

        # Create VAD instance
        self.vad = webrtcvad.Vad(1)  # can set vad from 0-3, with 0 only detecting louder speech

        # Buffer for microphone data
        self.x_len = 40000
        self.no_of_mics = 4
        self.input_mics = np.zeros((self.x_len, self.no_of_mics))


    """ take in microphone data (left ear, right ear, tail) and perform VAD check """
    def process_data(self, left_ear_data, right_ear_data, tail_data):
        # Perform VAD on each of the mic inputs
        is_speech_left = self.is_speech(left_ear_data)
        is_speech_right = self.is_speech(right_ear_data)
        is_speech_tail = self.is_speech(tail_data)

        # If any of the microphones detect speech, estimate the direction to that speech
        if is_speech_left or is_speech_right or is_speech_tail:
            print("Speech detected!")
            t1, t2 = self.estimate_direction()
            return t1, t2
        else:
            print("No speech detected.")
            return None, None


    """ check for speech in the audio frames """
    def is_speech(self, audio_data):
        # Process the audio in small frames for VAD
        frame_length = int(self.sample_rate * self.frame_duration_ms / 1000)  # Convert to samples
        audio_frames = self.block_data(audio_data, frame_size=frame_length)

        for frame in audio_frames:
            # VAD needs the audio to be in 16-bit mono PCM format
            if self.vad.is_speech(frame.tobytes(), self.sample_rate):
                return True
        return False


    @staticmethod
    # Split audio into smaller frames (blocks) to pass to VAD
    def block_data(data, block_size=500):
        num_of_blocks = len(data) // block_size
        blocks = []
        for i in range(num_of_blocks):
            start = i * block_size
            end = start + block_size
            blocks.append(data[start:end])
        return np.array(blocks)
    

    def turn_to_sound(self): 
      if self.audio_event[0] is None:
          return
      print("angular in degrees:{:.2f}".format(self.audio_event[0].ang))
      v = self.audio_event[0].azim
      # MiRo finish its rotation in 0.5s
      Tf = 0.5
      T1 = 0
      while(T1 <= Tf):
          # Rotate the robot to the direction of the sound
          self.msg_wheels.twist.linear.x = 0.0
          self.msg_wheels.twist.angular.z = v * 2
          self.pub_wheels.publish(self.msg_wheels)
          time.sleep(0.02)
          T1 += 0.02

      # After turning, drive forward towards the sound
      self.drive_toward_sound(v)


    def estimate_direction(self):
        # Direction estimation logic: Can be based on audio delays between microphones
        direction_x = 0
        direction_y = 0
        print(f"Estimated Direction: X: {direction_x}, Y: {direction_y}")
        return direction_x, direction_y
    

    def drive_toward_sound(self, azimuth):
        """
        Drive the robot towards the sound source.
        This method will continue to drive forward after orienting towards the sound.
        """
        # Adjust speed based on the azimuth or distance from the sound source if needed
        forward_speed = 0.1  # Forward speed (m/s)
        self.drive(forward_speed, forward_speed)

    def lock_onto_sound(self, ae_head):
      # Detect if it is the frame within the same event
      if ae_head.x == self.frame_p:
          self.status_code = 0  # Stop moving if it is the same event
      else:
          # The frame is different: not from the same event
          self.frame_p = ae_head.x
          self.turn_to_sound()  # Turn to the sound direction
          self.status_code = 0  # Continue to lock and drive towards sound


    def loop(self):
      msg_wheels = TwistStamped()

      # This switch loops through MiRo behaviours:
      # Listen to sound, turn to the sound source, and drive toward it
      self.status_code = 0
      while not rospy.core.is_shutdown():
          # Step 1: Sound event detection
          if self.status_code == 1:
              self.voice_accident()

          # Step 2: Orient towards it
          elif self.status_code == 2:
              self.lock_onto_sound(self.frame)
              # Clear the data collected when MiRo is turning
              self.audio_event = []

          # Fall back (initial detection state)
          else:
              self.status_code = 1


    def callback_mics(self, data):
      # Process the audio data for angular calculation
      self.audio_event = AudioEng.process_data(data.data)

      # Process data for dynamic thresholding and sound localization
      data_t = np.asarray(data.data, 'float32') * (1.0 / 32768.0)
      data_t = data_t.reshape((4, 500))
      self.head_data = data_t[2][:]
      if self.tmp is None:
          self.tmp = np.hstack((self.tmp, np.abs(self.head_data)))
      elif (len(self.tmp) < 10500):
          self.tmp = np.hstack((self.tmp, np.abs(self.head_data)))
      else:
          self.tmp = np.hstack((self.tmp[-10000:], np.abs(self.head_data)))
          self.thresh = self.thresh_min + AudioEng.non_silence_thresh(self.tmp)

      # Display data for debugging
      data = np.asarray(data.data)
      data = np.transpose(data.reshape((self.no_of_mics, 500)))
      data = np.flipud(data)
      self.input_mics = np.vstack((data, self.input_mics[:self.x_len - 500, :]))



if __name__ == "__main__":

    rospy.init_node("point_to_sound", anonymous=True)
    AudioEng = DetectAudioEngine()
    main = AudioClient()
    plt.show() # to stop signal display next run: comment this line and line 89(self.ani...)
    main.loop()