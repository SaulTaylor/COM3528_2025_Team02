from detect_emotion_client import EmotionClient
from comforting import Comforting
from miro_audio import SimpleAudioRecorder
import time

import rospy


if __name__ == "__main__":
    rospy.init_node("emotion_and_comforting_node", anonymous=True)
    #
    audio_client = SimpleAudioRecorder()
    rospy.spin()
    em_client = EmotionClient()
    ac_client = Comforting(3)
    # rospy.spin()
    while True:
        rospy.sleep(5)
        #audio_client.save_audio()


        print("about to call server")
        f = em_client.send_wav("miro_audio.wav")
        print(f)
        
        
        ac_client.angryAction(10)
        print("Performed action")
        rospy.sleep(2)

