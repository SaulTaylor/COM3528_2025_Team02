#!/usr/bin/env python3

import rospy
import actionlib

from com3528.msg import DetectEmotionAction, DetectEmotionGoal, DetectEmotionFeedback

class EmotionClient:
    # Initiates the client.
    def __init__(self):
        # Init node only if not already done
        if not rospy.core.is_initialized():
            rospy.init_node("emotion_client_node", anonymous=True)
            
        self.client = actionlib.SimpleActionClient("detect_emotion", DetectEmotionAction)
        rospy.loginfo("Waiting for emotion detection server...")
        self.client.wait_for_server()
        rospy.loginfo("Connected to server.")

    # Sends the .wav file to the server as bytes.
    def send_wav(self, wav_path):
        goal = DetectEmotionGoal()

        # Read .wav file as bytes
        with open(wav_path, "rb") as f:
            wav_data = f.read()
        # Convert bytes to uint8[]
        goal.audio_data = list(wav_data) 

        self.client.send_goal(goal, feedback_cb=self.feedback_cb)
        self.client.wait_for_result()

        result = self.client.get_result()
        rospy.loginfo(f"Detected Emotion: {result.detected_emotion}")

    def feedback_cb(self, feedback: DetectEmotionFeedback):
        rospy.loginfo(f"Feedback: {feedback.current_status}")

# Method to call to run the client.
def get_emotion(wav_path):
    client = EmotionClient()
    # Update this path
    client.send_wav(wav_path) 
