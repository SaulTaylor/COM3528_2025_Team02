#!/usr/bin/env python3

import rospy
import actionlib
import os

from com3528.msg import DetectEmotionAction, DetectEmotionGoal, DetectEmotionFeedback

class EmotionClient:
    def __init__(self):
        self.client = actionlib.SimpleActionClient("detect_emotion", DetectEmotionAction)
        rospy.loginfo("Waiting for emotion detection server...")
        self.client.wait_for_server()
        rospy.loginfo("Connected to server.")

    def send_wav(self, wav_path):
        goal = DetectEmotionGoal()

        # Read .wav file as bytes
        with open(wav_path, "rb") as f:
            wav_data = f.read()

        goal.audio_data = list(wav_data)  # Convert bytes to uint8[]

        self.client.send_goal(goal, feedback_cb=self.feedback_cb)
        self.client.wait_for_result()

        result = self.client.get_result()
        rospy.loginfo(f"Detected Emotion: {result.detected_emotion}")

    def feedback_cb(self, feedback: DetectEmotionFeedback):
        rospy.loginfo(f"Feedback: {feedback.current_status}")

if __name__ == "__main__":
    rospy.init_node("detect_emotion_client")

    # Dynamically find the .wav file next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wav_path = os.path.join(script_dir, "testAudio.wav")

    client = EmotionClient()
    client.send_wav(wav_path)  # <-- Update this path
