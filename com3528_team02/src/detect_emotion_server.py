#!/usr/bin/env python3

import rospy
import actionlib
import tempfile

from com3528_2.msg import DetectEmotionAction, DetectEmotionFeedback, DetectEmotionResult
from Models.wave_model.w2v2.miro_model_one import load_model, run_model

class EmotionServer:
    # Initiates the server.
    def __init__(self):
        self.server = actionlib.SimpleActionServer(
            "detect_emotion", DetectEmotionAction, self.execute_cb, False
        )
        self.server.start()
        print("running load")
        self.model = load_model()
        rospy.loginfo("Emotion detection server is ready.")

    # Gets the audio file and runs the model to detemrine an emotion.
    def execute_cb(self, goal):
        feedback = DetectEmotionFeedback()
        result = DetectEmotionResult()

        rospy.loginfo("Received audio data.")
        feedback.current_status = "Saving audio data..."
        self.server.publish_feedback(feedback)

        # Saves the audio file to the current working directory.
        # Audio is saved in a temporary file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(bytearray(goal.audio_data))
            audio_path = f.name

        rospy.loginfo(f"Saved audio to: {audio_path}")
        feedback.current_status = "Running emotion detection..."
        self.server.publish_feedback(feedback)

        # Stub detection logic
        detected_emotion = self.detect_emotion(audio_path)

        feedback.current_status = "Complete"
        self.server.publish_feedback(feedback)
        # Sets the detected emotion as the result and returns it.
        result.detected_emotion = detected_emotion
        self.server.set_succeeded(result)
        
    # Calls the model to determine an emotion.
    def detect_emotion(self, path):
        prediction = run_model(path, self.model)
        rospy.loginfo(f"Analyzing {path}...")
        rospy.loginfo(prediction)
        return prediction

if __name__ == "__main__":
    rospy.init_node("detect_emotion_server")
    server = EmotionServer()
    rospy.spin()
