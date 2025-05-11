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

    def __init__(self, action_duration):
        """
        Class initialisation
        """

        self.ACTION_DURATION = rospy.Duration(action_duration)  # Set the action duration dynamically
        self.TICK = 1  # Main loop frequency (in secs)

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

        # List of action functions
        ##NOTE Try writing your own action functions and adding them here
        self.actions = [
            self.earWiggle,
            self.tailWag,
            self.rotate,
            self.raiseHead,
            self.happyAction,
            self.sadAction,
            self.angryAction,
            self.tail
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
            self.head,
            self.tail,
        ) = range(8)

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
    
    def raiseHead(self,):
            print("MiRo raising head")
            A = 1.0
            w = 2 * np.pi * 0.2
            f = lambda t: A * np.cos(w * t)
            i = 0
            t0 = rospy.Time.now()
            while rospy.Time.now() < t0 + self.ACTION_DURATION:
                self.cos_joints.data[self.head] = f(i)
                self.pub_cos.publish(self.cos_joints)
                i += self.TICK
                rospy.sleep(self.TICK)
            self.cos_joints.data[self.head] = 0.0
            self.pub_cos.publish(self.cos_joints)

    def tailWag(self, t0):
            print("MiRo wagging tail")
            A = 1.0
            w = 2 * np.pi * 0.2 # Change depending on how fast tail wags
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
                self.velocity.twist.angular.z = 0.9 # How fast the miro rotates
                self.pub_cmd_vel.publish(self.velocity)
            self.velocity.twist.linear.x = 0
            self.velocity.twist.angular.z = 0
            self.pub_cmd_vel.publish(self.velocity)

    def happyAction(self, duration):
        print("Emotion Received: Happiness")
        t0 = rospy.Time.now()
        end_time = t0 + rospy.Duration(duration)

        A = 1.0
        w = 2 * np.pi * 1  # 1 Hz = 1 wag/sec
        f = lambda t: A * np.cos(w * t)
        i = 0

        while rospy.Time.now() < end_time:

            self.cos_joints.data[self.wag] = f(i)

            if i % 5 == 0:
                self.cos_joints.data[self.head] = 1.0
            elif i % 5 == 2:
                self.cos_joints.data[self.head] = 0.0

            if i % 7 == 0:
                self.cos_joints.data[self.left_ear] = 1.0
                self.cos_joints.data[self.right_ear] = -1.0
            elif i % 7 == 2:
                self.cos_joints.data[self.left_ear] = 0.0
                self.cos_joints.data[self.right_ear] = 0.0

            if i % 6 == 0:
                self.velocity.twist.angular.z = 0.5
            else:
                self.velocity.twist.angular.z = 0.0

            self.pub_cos.publish(self.cos_joints)
            self.pub_cmd_vel.publish(self.velocity)

            i += self.TICK
            rospy.sleep(self.TICK)

        self.cos_joints.data[self.wag] = 0.0
        self.cos_joints.data[self.head] = 0.0
        self.cos_joints.data[self.left_ear] = 0.0
        self.cos_joints.data[self.right_ear] = 0.0
        self.velocity.twist.angular.z = 0.0
        self.pub_cos.publish(self.cos_joints)
        self.pub_cmd_vel.publish(self.velocity)
        
    def fearAndSadnessAction(self, duration):

        print("Emotion Received: Fear")
        t0 = rospy.Time.now()
        A = 1.0
        w = 2 * np.pi * 0.2
        f = lambda t: A * np.cos(w * t)
        i = 0

        while rospy.Time.now() < t0 + rospy.Duration(duration):
            self.cos_joints.data[self.head] = -1.0
            self.cos_joints.data[self.tail] = -1.0
            self.velocity.twist.linear.x = 0
            self.velocity.twist.angular.z = 0.7 # How fast the miro rotates
            self.pub_cmd_vel.publish(self.velocity)
            self.pub_cos.publish(self.cos_joints) 
            i += self.TICK  
            rospy.sleep(self.TICK)
        
        self.cos_joints.data[self.head] = 0.0
        self.cos_joints.data[self.tail] = 0.0
    
    def angryAction(self, duration):
         
        print("Emotion Received: Anger")
        t0 = rospy.Time.now()
        A = 1.0
        w = 2 * np.pi * 0.2
        f = lambda t: A * np.cos(w * t)
        i = 0

        while rospy.Time.now() < t0 + rospy.Duration(duration):
            self.cos_joints.data[self.head] = -1.0
            self.cos_joints.data[self.tail] = -1.0  
            i += self.TICK  
            rospy.sleep(self.TICK)
        
        self.cos_joints.data[self.head] = 0.0
        self.cos_joints.data[self.tail] = 0.0
    
    def calmAndNeutralAction(self, duration):
        print("Emotion Received: Calm")
        t0 = rospy.Time.now()
        A = 1.0
        w = 2 * np.pi * 0.2
        f = lambda t: A * np.cos(w * t)
        i = 0

        while rospy.Time.now() < t0 + rospy.Duration(duration):
            self.cos_joints.data[self.earWiggle]
            if i % 10 == 0:
                self.cos_joints.data[self.yaw] = 1.0  # right
            elif i % 10 == 5:
                self.cos_joints.data[self.yaw] = -1.0  # left
            elif i % 10 == 7:
                self.cos_joints.data[self.yaw] = 0.0  # return to center
            self.pub_cos.publish(self.cos_joints)

            i += self.TICK
            rospy.sleep(self.TICK)
        
        self.cos_joints.data[self.wag] = 0.0
        self.cos_joints.data[self.yaw] = 0.0
        self.pub_cos.publish(self.cos_joints)

             
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
