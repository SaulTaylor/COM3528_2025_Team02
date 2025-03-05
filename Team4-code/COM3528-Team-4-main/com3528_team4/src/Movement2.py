#!/usr/bin/env python3

import os
import numpy as np
import rospy
import miro2 as miro
import geometry_msgs
from AudioEngine import DetectAudioEngine
from std_msgs.msg import Int16MultiArray
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from geometry_msgs.msg import Twist, TwistStamped
import time
from miro2.lib import wheel_speed2cmd_vel


class Movement:
    def __init__(self):
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

       

        # publishers
        self.pub_push = rospy.Publisher(topic_base_name + "/core/mpg/push", miro.msg.push, queue_size=0)
        self.pub_wheels = rospy.Publisher(topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0)

        # prepare push message
        self.msg_push = miro.msg.push()
        self.msg_push.link = miro.constants.LINK_HEAD
        self.msg_push.flags = (miro.constants.PUSH_FLAG_NO_TRANSLATION + miro.constants.PUSH_FLAG_VELOCITY)

        # status flags
        self.audio_event = None
        self.orienting = False
        self.action_time = 1  # secs
        self.thresh = 0.05

        # time
        self.frame_p = None
        self.msg_wheels = TwistStamped()
        self.controller = miro.lib.PoseController()
        self.cmd_vel = miro.lib.DeltaPose()

    def drive(self, speed_l=0.1, speed_r=0.1):
        wheel_speed = [speed_l, speed_r]

        # convert to comand vel from wheel speed
        (dr, dtheta) = wheel_speed2cmd_vel(wheel_speed)

        # Update the message with the desired speed
        self.msg_wheels.twist.linear.x = dr
        self.msg_wheels.twist.angular.z = dtheta

        # Publish message to control/cmd_vel topic
        self.pub_wheels.publish(self.msg_wheels)

    def turn_to_sound(self, azimuth):
        # Turn to the sound source
        tf = 2
        t0 = 0
        counter = 0
        while t0 <= tf:
            counter += 1
            print(counter)

            # self.drive(v*2,v*2)
            self.msg_wheels.twist.linear.x = 0.0
            self.msg_wheels.twist.angular.z = azimuth /2

            # test output
            # self.msg_wheels.twist.angular.z = 0.0

            self.pub_wheels.publish(self.msg_wheels)
            rospy.sleep(0.01)
            t0 += 0.01


    def loop(self):
        print("loop")
        self.turn_to_sound(np.deg2rad(-180))
        

if __name__ == '__main__':
    print("Initialising")
    rospy.init_node('mov')
    mov = Movement()
    mov.loop()

