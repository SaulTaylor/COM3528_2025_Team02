cmake_minimum_required(VERSION 3.0.2)
project(com3528)

## Find required packages
find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  std_msgs
  message_generation
  rospy
)

## Declare action files
add_action_files(
  DIRECTORY action
  FILES DetectEmotion.action
)

## Generate messages
generate_messages(
  DEPENDENCIES std_msgs actionlib_msgs
)

## Declare the catkin package
catkin_package(
  CATKIN_DEPENDS actionlib_msgs std_msgs message_runtime
)

catkin_install_python(PROGRAMS
  src/basic_audio_text.py
  src/detect_emotion_server.py
  src/comforting.py
  src/train_classifier.py
  src/detect_emotion_client.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
