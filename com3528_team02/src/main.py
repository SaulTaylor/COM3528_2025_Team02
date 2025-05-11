from detect_emotion_client import EmotionClient
from comforting import Comforting
import time

import rospy


if __name__ == "__main__":
    rospy.init_node("emotion_and_comforting_node", anonymous=True)
    em_client = EmotionClient()
    ac_client = Comforting(5)

    while True:
        em_client.send_wav("testAudio.wav")
        ac_client.angryAction(10)
        time.sleep(5)

