#!/usr/bin/env python3

import rospy
from std_msgs.msg import UInt16MultiArray, Int16MultiArray

import time
import sys
import os
import numpy as np
import wave, struct

import miro2 as miro

import soundfile as sf
from vosk import Model, KaldiRecognizer

# amount to keep the buffer stuffed - larger numbers mean
# less prone to dropout, but higher latency when we stop
# streaming. with a read-out rate of 8k, 4000 samples will
# buffer for half of a second, for instance.
BUFFER_STUFF_SAMPLES = 4000

# messages larger than this will be dropped by the receiver,
# however, so - whilst we can stuff the buffer more than this -
# we can only send this many samples in any single message.
MAX_STREAM_MSG_SIZE = (4096 - 48)

# using a margin avoids sending many small messages - instead
# we will send a smaller number of larger messages, at the cost
# of being less precise in respecting our buffer stuffing target.
BUFFER_MARGIN = 1000
BUFFER_MAX = BUFFER_STUFF_SAMPLES + BUFFER_MARGIN
BUFFER_MIN = BUFFER_STUFF_SAMPLES - BUFFER_MARGIN

# how long to record before playing back in seconds?
RECORD_TIME = 2

# microphone sample rate (also available at miro2.constants)
MIC_SAMPLE_RATE = 20000

# sample count
SAMPLE_COUNT = RECORD_TIME * MIC_SAMPLE_RATE



################################################################

def error(msg):
	print(msg)
	sys.exit(0)

class Client:
    def __init__(self, mode=None):
        # Create robot interface
        self.interface = miro.lib.RobotInterface()

        # State variables
        self.micbuf = np.zeros((0, 4), 'uint16')
        self.buffer_stuff = 0

        # Load Vosk model for speech recognition
        model_path = "model"  # Ensure you download a Vosk model and set this path
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)  # Assuming 16kHz audio

        # Robot name
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

        # Subscribe to stream topic
        topic = topic_base_name + "/sensors/stream"
        print("subscribe", topic)
        self.sub_stream = rospy.Subscriber(topic, UInt16MultiArray, self.callback_stream, queue_size=1, tcp_nodelay=True)

        # Subscribe to microphones
        self.interface.register_callback("microphones", self.callback_mics)

        # Report
        print("Recording")

    def callback_stream(self, msg):
        self.buffer_space = msg.data[0]
        self.buffer_total = msg.data[1]
        self.buffer_stuff = self.buffer_total - self.buffer_space

    def callback_mics(self, msg):
        self.micbuf = np.concatenate((self.micbuf, msg.data))

        # Convert uint16 data to int16 PCM format
        pcm_audio = self.micbuf.astype(np.int16).tobytes()

        # Process audio with Vosk
        if self.recognizer.AcceptWaveform(pcm_audio):
            result = self.recognizer.Result()
            print("Recognized Speech:", result)

        # Report
        sys.stdout.write(".")
        sys.stdout.flush()

    def loop(self):
        while not rospy.core.is_shutdown():
            time.sleep(0.02)
        self.interface.disconnect()


if __name__ == "__main__":
    main = Client()
    main.loop()
