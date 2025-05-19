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


    emotion_labels = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }

    map_dic = {
        'neutral': ac_client.calmAndNeutralAction,
        'calm': ac_client.calmAndNeutralAction,
        'happy': ac_client.happyAction,
        'sad': ac_client.sadAction,
        'angry': ac_client.angryAction,
        'fearful': ac_client.fearAction,
        'disgust': ac_client.angryAction,
        'surprised': ac_client.happyAction
    }

    audio_files = ["angry.wav"]

    for audio in audio_files:

        time.sleep(1)

        result = em_client.send_wav(audio)

        if result in map_dic:
            print(result)
            map_dic["happy"](10)  # Calls handle_angry()
        else:
            print("No handler for this emotion.")

        time.sleep(7)

