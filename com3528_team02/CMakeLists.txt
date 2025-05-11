cmake_minimum_required(VERSION 3.0.2)
project(com3528_2)

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