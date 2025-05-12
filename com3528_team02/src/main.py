from detect_emotion_client import EmotionClient
from comforting import Comforting
from miro_audio import SimpleAudioRecorder
import time

import rospy


if __name__ == "__main__":
    rospy.init_node("emotion_and_comforting_node", anonymous=True)
    #
    em_client = EmotionClient()
    ac_client = Comforting(3)

    ac_client.calmAndNeutralAction(10)

