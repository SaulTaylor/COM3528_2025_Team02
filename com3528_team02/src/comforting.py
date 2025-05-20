#!/usr/bin/env python3

import random
import os
import numpy as np
import rospy
import miro2 as miro
from geometry_msgs.msg import Twist, TwistStamped
import time
import math
from sensor_msgs.msg import JointState
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
        self.pub_kin = rospy.Publisher(
             topic_root + "/control/kinematic_joints", JointState, queue_size=0
        )

        # List of action functions
        self.actions = [
            self.earWiggle,
            self.tailWag,
            self.rotate,
            self.raiseHead,
            self.happyAction,   
            self.angryAction,
            self.sadAction,
            self.calmAndNeutralAction,
            self.fearAction,
        ]

        # Initialise objects for data storage and publishing
        self.light_array = None
        self.velocity = TwistStamped()
        self.cos_joints = Float32MultiArray()
        self.cos_joints.data = [0.0] * 6
        self.kin_joints = JointState()
        self.kin_joints.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin_joints.position = [0.0, math.radians(34.0), 0.0, 0.0]
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

        self.pub_cos.publish(self.cos_joints)
        self.pub_kin.publish(self.kin_joints)

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
    
    def raiseHead(self, t0):
            print("MiRo raising head")
            self.kin_joints.position[self.pitch] = 0
            self.pub_kin.publish(self.kin_joints)
            A = 1.0
            w = 2 * np.pi * 0.2
            f = lambda t: A * np.cos(w * t)
            i = 0
            t0 = rospy.Time.now()
            while rospy.Time.now() < t0 + self.ACTION_DURATION:
                self.kin_joints.position[self.pitch] = 1
                self.pub_kin.publish(self.kin_joints)
                i += self.TICK
                rospy.sleep(self.TICK)
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
        start_time = rospy.Time.now()
        end_time = start_time + rospy.Duration(duration)
        rate = rospy.Rate(20)

        current_yaw = 0.0
        self.kin_joints.position[self.yaw] = current_yaw
        self.pub_kin.publish(self.kin_joints)

        next_change_secs = random.uniform(0.5, 2.0)
        last_change_time = rospy.Time.now()

        A = 1.0
        w = 2 * np.pi * 0.2
        f = lambda t: A * np.cos(w * t)
        i = 0

        while rospy.Time.now() < end_time:
            now = rospy.Time.now()
            elapsed = (now - last_change_time).to_sec()
            self.velocity.twist.angular.z = 0.2
            self.pub_cmd_vel.publish(self.velocity)
            self.kin_joints.position[self.pitch] = -1.0
            self.kin_joints.position[self.lift] = -1.0
            self.pub_kin.publish(self.kin_joints)
            if elapsed >= next_change_secs:
                current_yaw = random.choice([-1.0, 0.0, 1.0])
                self.kin_joints.position[self.yaw] = current_yaw
                self.pub_kin.publish(self.kin_joints)

                next_change_secs = random.uniform(0.5, 2.0)
                last_change_time = now

            self.tailWag(now)
            self.earWiggle(now)

            rate.sleep()

        # Reset to neutral at the end
        self.kin_joints.position[self.yaw] = 0.0
        self.kin_joints.position[self.pitch] = 0.0
        self.kin_joints.position[self.lift] = 0.0
        self.cos_joints.data[self.wag] = 0.0
        self.cos_joints.data[self.left_ear] = 0.0
        self.cos_joints.data[self.right_ear] = 0.0
        self.pub_cos.publish(self.cos_joints)
        self.pub_cmd_vel.publish(self.velocity)
        self.pub_kin.publish(self.kin_joints)
        
    def fearAction(self, duration):

        print("Emotion Received: Fear")
        t0 = rospy.Time.now()
        A = 1.0
        w = 2 * np.pi * 0.2
        f = lambda t: A * np.cos(w * t)
        i = 0
        rate = rospy.Rate(20)

        while rospy.Time.now() < t0 + rospy.Duration(duration):
            self.kin_joints.position[self.pitch] = 1.0
            self.kin_joints.position[self.lift] = 1.0
            self.cos_joints.data[self.droop] = 1.0
            self.velocity.twist.linear.x = 0
            self.velocity.twist.angular.z = 0.2 # How fast the miro rotates
            self.pub_cmd_vel.publish(self.velocity)
            self.pub_cos.publish(self.cos_joints) 
            self.pub_kin.publish(self.kin_joints)
            i += self.TICK  
            rate.sleep()
        
        self.kin_joints.position[self.pitch] = 0.0
        self.kin_joints.position[self.lift] = 0.0
        self.cos_joints.data[self.wag] = 0.0
        self.velocity.twist.linear.x = 0.0
        self.velocity.twist.angular.z = 0.0 
        self.pub_cmd_vel.publish(self.velocity)
        self.pub_kin.publish(self.kin_joints)
        self.pub_cos.publish(self.cos_joints)
    
    def sadAction(self, duration):
        print("Emotion Received: Sadness")
        t0 = rospy.Time.now()
        A = 1.0
        w = 2 * np.pi * 0.2
        f = lambda t: A * np.cos(w * t)
        i = 0
        rate = rospy.Rate(20)

        while rospy.Time.now() < t0 + rospy.Duration(duration):
            self.kin_joints.position[self.pitch] = -1.0
            self.kin_joints.position[self.lift] = -1.0
            self.cos_joints.data[self.droop] = -1.0
            self.cos_joints.data[self.left_ear] = 1.0
            self.cos_joints.data[self.right_ear] = 1.0
            if i % 6 == 0:
                self.velocity.twist.angular.z = 0.5
            else:
                self.velocity.twist.angular.z = 0.0
            self.pub_cmd_vel.publish(self.velocity)
            self.pub_kin.publish(self.kin_joints)
            self.pub_cos.publish(self.cos_joints) 
            i += self.TICK  
            rospy.sleep(self.TICK)
        
        self.kin_joints.position[self.pitch] = 0.0
        self.kin_joints.position[self.lift] = 0.0
        self.cos_joints.data[self.wag] = 0.0
        self.pub_kin.publish(self.kin_joints)
        self.pub_cos.publish(self.cos_joints)

    
    def angryAction(self, duration):
        print("Emotion Received: Anger")
        t0 = rospy.Time.now()
        A = 1.0  # maximum value
        T_total = 10.0  # total duration for the linear increase

        f = lambda t: (A / T_total) * t if t <= T_total else A
        i = 0
        rate = rospy.Rate(20)

        while rospy.Time.now() < t0 + rospy.Duration(duration):
            self.kin_joints.position[self.pitch] = -1.0
            self.kin_joints.position[self.lift] = -1.0
            self.pub_kin.publish(self.kin_joints)
            i += self.TICK  
            rate.sleep()

        self.cos_joints.data[self.droop] = 0.0
        self.kin_joints.position[self.lift] = 0.0
        self.cos_joints.data[self.wag] = 0.0
        self.kin_joints.position[self.pitch] = 0.0
        self.pub_kin.publish(self.kin_joints)
        self.pub_cos.publish(self.cos_joints)
    
    def calmAndNeutralAction(self, duration):
        print("Emotion Received: Calm")
        start_time = rospy.Time.now()
        end_time = start_time + rospy.Duration(duration)
        rate  = rospy.Rate(20)

        current_yaw = 0.0
        self.kin_joints.position[self.yaw] = current_yaw
        self.pub_kin.publish(self.kin_joints)

        next_change_secs = random.uniform(0.5, 2.0)
        last_change_time = rospy.Time.now()

        while rospy.Time.now() < end_time:
            now = rospy.Time.now()
            elapsed = (now - last_change_time).to_sec()
            if elapsed >= next_change_secs:
                current_yaw = random.choice([-1.0, 0.0, 1.0])
                self.kin_joints.position[self.yaw] = current_yaw
                self.pub_kin.publish(self.kin_joints)

                next_change_secs = random.uniform(0.5, 2.0)
                last_change_time = now

            self.earWiggle(now)

            rate.sleep()

        self.kin_joints.position[self.yaw] = 0.0
        self.pub_kin.publish(self.kin_joints)
        # clear any COS controls (if needed)
        self.cos_joints.data[self.wag] = 0.0
        self.cos_joints.data[self.left_ear] = 0.0
        self.cos_joints.data[self.right_ear] = 0.0
        self.pub_cos.publish(self.cos_joints)



# This is run when the script is called directly
if __name__ == "__main__":
    rospy.init_node("comforting_runner", anonymous=True)
    duration_seconds = rospy.get_param("~action_duration", 5)
    main = Comforting(duration_seconds)  # Instantiate class
