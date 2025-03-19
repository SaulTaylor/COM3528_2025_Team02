#!/usr/bin/env python3

import os
import numpy as np
import rospy
import miro2 as miro
from geometry_msgs.msg import Twist, TwistStamped
import time
from miro2.lib import wheel_speed2cmd_vel
from std_msgs.msg import (
    Float32MultiArray,
    UInt32MultiArray,
    UInt16,
)  # Used in callbacks

class Comforting:

    TICK = 0.02  # Main loop frequency (in secs, default is 50Hz)
    ACTION_DURATION = rospy.Duration(3.0)  # seconds

    def __init__(self):
        """
        Class initialisation
        """
        print("Initialising the controller...")

        # Get robot name
        topic_root = "/" + os.getenv("MIRO_ROBOT_NAME")

        # Initialise a new ROS node to communicate with MiRo
        if not rospy.is_shutdown():
            rospy.init_node("comforting", anonymous=True)

        # Define ROS publishers
        self.pub_cmd_vel = rospy.Publisher(
            topic_root + "/control/cmd_vel", TwistStamped, queue_size=0
        )
        self.pub_cos = rospy.Publisher(
            topic_root + "/control/cosmetic_joints", Float32MultiArray, queue_size=0
        )
        self.pub_illum = rospy.Publisher(
            topic_root + "/control/illum", UInt32MultiArray, queue_size=0
        )

        # Define ROS subscribers
        rospy.Subscriber(
            topic_root + "/sensors/touch_head",
            UInt16,
            self.touchHeadListener,
        )
        rospy.Subscriber(
            topic_root + "/sensors/touch_body",
            UInt16,
            self.touchBodyListener,
        )
        rospy.Subscriber(
            topic_root + "/sensors/light",
            Float32MultiArray,
            self.lightCallback,
        )

        # List of action functions
        ##NOTE Try writing your own action functions and adding them here
        self.actions = [
            self.earWiggle,
            self.tailWag,
            self.rotate,
        ]

        # Initialise objects for data storage and publishing
        self.light_array = None
        self.velocity = TwistStamped()
        self.cos_joints = Float32MultiArray()
        self.cos_joints.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.illum = UInt32MultiArray()
        self.illum.data = [
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
        ]

        # Utility enums
        self.tilt, self.lift, self.yaw, self.pitch = range(4)
        (
            self.droop,
            self.wag,
            self.left_eye,
            self.right_eye,
            self.left_ear,
            self.right_ear,
        ) = range(6)

        # Variables for Q-learning algorithm
        self.reward = 0
        self.punishment = 0
        self.Q = [0] * len(self.actions)  # Highest Q value gets to run
        self.N = [0] * len(self.actions)  # Number of times an action was done
        self.r = 0  # Current action index
        self.alpha = 0.7  # learning rate
        self.discount = 25  # discount factor (anti-damping)

        # Give it a sec to make sure everything is initialised
        rospy.sleep(1.0)  

    def earWiggle(self, t0):
            print("MiRo wiggling ears")
            A = 1.0
            w = 2 * np.pi * 0.2
            f = lambda t: A * np.cos(w * t)
            i = 0
            while rospy.Time.now() < t0 + self.ACTION_DURATION:
                self.cos_joints.data[self.left_ear] = f(i)
                self.cos_joints.data[self.right_ear] = f(i)
                self.pub_cos.publish(self.cos_joints)
                i += self.TICK
                rospy.sleep(self.TICK)
            self.cos_joints.data[self.left_ear] = 0.0
            self.cos_joints.data[self.right_ear] = 0.0
            self.pub_cos.publish(self.cos_joints)

    def tailWag(self, t0):
            print("MiRo wagging tail")
            A = 1.0
            w = 2 * np.pi * 0.2
            f = lambda t: A * np.cos(w * t)
            i = 0
            while rospy.Time.now() < t0 + self.ACTION_DURATION:
                self.cos_joints.data[self.wag] = f(i)
                self.pub_cos.publish(self.cos_joints)
                i += self.TICK
                rospy.sleep(self.TICK)
            self.cos_joints.data[self.wag] = 0.0
            self.pub_cos.publish(self.cos_joints)

    def rotate(self, t0):
            print("MiRo rotating")
            while rospy.Time.now() < t0 + self.ACTION_DURATION:
                self.velocity.twist.linear.x = 0
                self.velocity.twist.angular.z = 0.2
                self.pub_cmd_vel.publish(self.velocity)
            self.velocity.twist.linear.x = 0
            self.velocity.twist.angular.z = 0
            self.pub_cmd_vel.publish(self.velocity)
    
    def loop(self):
        """
        Main loop
        """
        print("Starting the loop")
        while not rospy.core.is_shutdown():
            # Select next action randomly or via Q score with equal probability
            
            print("Performing random action")
            self.r = np.random.randint(0, len(self.actions))

            # Run the selected action and update the action counter N accordingly
            start_time = rospy.Time.now()
            self.actions[self.r](start_time)



# This is run when the script is called directly
if __name__ == "__main__":
    main = Comforting()  # Instantiate class
    main.loop()  # Run the main control loop
