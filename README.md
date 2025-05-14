# COM3528---Team-02

This code, despite our best efforts, is probably not reproducible but the instructions are here

### INSTALLATIONS
We dont have a requirements file - if you want to test the code you will need to manually install each package.

### Installation
Download Audio_Speech_Actors_01-24.zip from https://zenodo.org/records/1188976, extract it to src/models/w2v2 and rename the path inside train classifier to the name of the extracted folder.

Then run train_classifier.py - some file paths might need to be changed, and the code will need to be run from src

Then run roscore, rosrun com3528_2 detect_emotion_server.py, and then python3 main.py in three terminal instances inside the src directory


