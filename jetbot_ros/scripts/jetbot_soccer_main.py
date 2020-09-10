#! /usr/bin/env python
############## ROS Import ###############
import rospy
import std_msgs
from sensor_msgs.msg import Image

import numpy as np
import random
import time
import itertools
import os

#import jetson.inference
#import jetson.utils

import cv2

from cv_bridge import CvBridge, CvBridgeError

#net = jetson.inference.detectNet("ssd-mobilenet-v2")

bridge = CvBridge()

lidar_value = 0
infrared_value = 0
def image_callback(msg):
    #print("image_callback")
    cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2RGBA).astype(np.float32)
    #print("cv_image.shape: " + str(cv_image.shape))
    #print("type(cv_image): " + str(type(cv_image)))
    
    width = 1280
    height = 720
    #cuda_image = jetson.utils.cudaFromNumpy(cv_image)
    #array = jetson.utils.cudaToNumpy(cuda_image, 720, 1280, 3)
    #jetson.utils.saveImageRGBA("cuda-from-numpy.jpg", cuda_image, 720, 1280)
    #print("cv_image.shape: " + str(cv_image.shape))
    #detections = net.Detect(cuda_image, width, height, "box,labels,conf")

    # print the detections
    #print("detected {:d} objects in image".format(len(detections)))

    #for detection in detections:
    #    print(detection)
    
    cv2.imwrite("array_image.jpg", cv_image)
    #jetson.utils.saveImageRGBA("detected_image.jpg", cv_image, width, height)
    
    
def lidar_callback(msg):
    global lidar_value
    lidar_value = msg
    
    #print("lidar: " + str(msg))
    

def infrared_callback(msg):
    global infrared_value
    
    infrared_value = msg
    #print("infrared: " + str(msg))
    
    
############## ROS Part ###############
rospy.init_node('deepsoccer')
wheel1 = rospy.Publisher('/deepsoccer_motors/cmd_str_wheel1', std_msgs.msg.String, queue_size=5)
wheel2 = rospy.Publisher('/deepsoccer_motors/cmd_str_wheel2', std_msgs.msg.String, queue_size=5)
wheel3 = rospy.Publisher('/deepsoccer_motors/cmd_str_wheel3', std_msgs.msg.String, queue_size=5)
wheel4 = rospy.Publisher('/deepsoccer_motors/cmd_str_wheel4', std_msgs.msg.String, queue_size=5)
solenoid = rospy.Publisher('/deepsoccer_solenoid/cmd_str', std_msgs.msg.String, queue_size=5)
roller = rospy.Publisher('/deepsoccer_roller/cmd_str', std_msgs.msg.String, queue_size=5)
rospy.Subscriber("/deepsoccer_camera/raw", Image, image_callback)
rospy.Subscriber("/deepsoccer_lidar", std_msgs.msg.String, lidar_callback)
rospy.Subscriber("/deepsoccer_infrared", std_msgs.msg.String, infrared_callback)
 
rate = rospy.Rate(2000)

stop_action = [0, 0, 0, 0, 'stop', 'out']
forward_action = [1074, 50, 50, 1074, 'in', 'out']
left_action = [50, 50, 50, 50, 'in', 'out']
right_action = [1074, 1074, 1074, 1074, 'in', 'out']
bacward_action = [50, 1074, 1074, 50, 'in', 'out']
hold_action = [0, 0, 0, 0, 'in', 'out']
kick_action = [0, 0, 0, 0, 'stop', 'in']
robot_action_list = [stop_action, forward_action, left_action, right_action, bacward_action, hold_action, kick_action]

############## ROS + Deep Learning Part ###############
while not rospy.is_shutdown():
    action_index = 0
    action = robot_action_list[action_index]
    print("action: " + str(action))
    print("lidar_value: " + str(lidar_value))
    print("infrared_value: " + str(infrared_value))
    print("")
    
    wheel1_action = action[0]
    wheel2_action = action[1]
    wheel3_action = action[2]
    wheel4_action = action[3]
    roller_action = action[4]
    solenoid_action = action[5]
    
    wheel1.publish(str(wheel1_action))
    wheel2.publish(str(wheel2_action))
    wheel3.publish(str(wheel3_action))
    wheel4.publish(str(wheel4_action))
    roller.publish(roller_action)
    solenoid.publish(solenoid_action)
    
    time.sleep(0.5)
 
rate.sleep()