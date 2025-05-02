#!/usr/bin/env python3

import rospy
import actionlib

from com3528.msg import DetectEmotionAction, DetectEmotionGoal, DetectEmotionFeedback

class EmotionClient:
    # Initiates the client.
    def __init__(self):
        # Init node only if not already done 
        print("initilaising")
        if not rospy.core.is_initialized():
            rospy.init_node("emotion_client_node", anonymous=True)
            
        self.client = actionlib.SimpleActionClient("detect_emotion", DetectEmotionAction)
        rospy.loginfo("Waiting for emotion detection server...")
        self.client.wait_for_server()
        rospy.loginfo("Connected to server.")

    # Sends the .wav file to the server as bytes.
    def send_wav(self, wav_path):
        print("1")
        goal = DetectEmotionGoal()
        print("2")
        import os
        # Read .wav file as bytes
        #path = os.path.join(os.path.expanduser("~"), "mdk")
        if not os.path.isfile(wav_path):
            rospy.logerr(f"file not found: {wav_path}")
        with open(wav_path, "rb") as f:
            wav_data = f.read()
        # Convert bytes to uint8[]
        goal.audio_data = list(wav_data) 

        self.client.send_goal(goal, feedback_cb=self.feedback_cb)
        print("goal sent")
        self.client.wait_for_result()

        result = self.client.get_result()
        rospy.loginfo(f"Detected Emotion: {result.detected_emotion}")

    def feedback_cb(self, feedback: DetectEmotionFeedback):
        rospy.loginfo(f"Feedback: {feedback.current_status}")

# Method to call to run the client.
def get_emotion(wav_path):
    client = EmotionClient()
    print("finished seetin up lcinegt")
    # Update this path
    client.send_wav(wav_path) 

if __name__ == "__main__":
    get_emotion("/home/student/mdk/catkin_ws/src/COM3528_2025_Team02/com3528/src/test_argument.wav")