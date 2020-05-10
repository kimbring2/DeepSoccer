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

import jetson.inference
import jetson.utils

import cv2

from cv_bridge import CvBridge, CvBridgeError

net = jetson.inference.detectNet("ssd-mobilenet-v2")

bridge = CvBridge()

def image_callback(msg):
    print("image_callback")
    cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2RGBA).astype(np.float32)
    #print("cv_image.shape: " + str(cv_image.shape))
    #print("type(cv_image): " + str(type(cv_image)))
    
    width = 1280
    height = 720
    cuda_image = jetson.utils.cudaFromNumpy(cv_image)
    #array = jetson.utils.cudaToNumpy(cuda_image, 720, 1280, 3)
    #jetson.utils.saveImageRGBA("cuda-from-numpy.jpg", cuda_image, 720, 1280)
    #print("cv_image.shape: " + str(cv_image.shape))
    detections = net.Detect(cuda_image, width, height, "box,labels,conf")

    # print the detections
    print("detected {:d} objects in image".format(len(detections)))

    for detection in detections:
        print(detection)
    
    #cv2.imwrite("array_image.jpg", array)
    jetson.utils.saveImageRGBA("detected_image.jpg", cuda_image, width, height)
    
    
############## ROS Part ###############
rospy.init_node('jetbot_soccer')

wheel1 = rospy.Publisher('/jetbot_soccer_motors/cmd_str_wheel1', std_msgs.msg.String, queue_size=5)
wheel2 = rospy.Publisher('/jetbot_soccer_motors/cmd_str_wheel2', std_msgs.msg.String, queue_size=5)
wheel3 = rospy.Publisher('/jetbot_soccer_motors/cmd_str_wheel3', std_msgs.msg.String, queue_size=5)
wheel4 = rospy.Publisher('/jetbot_soccer_motors/cmd_str_wheel4', std_msgs.msg.String, queue_size=5)
rospy.Subscriber("/jetbot_camera/raw", Image, image_callback)
 
rate = rospy.Rate(2000)

stop_action = [0, 0, 0, 0, 30, 80]
forward_action = [1074, 50, 50, 1074, 30, 0]
left_action = [50, 50, 50, 50, 50, 0]
right_action = [1074, 1074, 1074, 1074, 50, 0]
bacward_action = [50, 1074, 1074, 50, 50, 0]
hold_action = [0, 0, 0, 0, 30, 0]
kick_action = [0, 0, 0, 0, 0, -80]
robot_action_list = [stop_action, forward_action, left_action, right_action, bacward_action, hold_action, kick_action,]

############## ROS + Deep Learning Part ###############
while not rospy.is_shutdown():
    action_index = 0
    action = robot_action_list[action_index]
    
    wheel1_action = action[0]
    wheel2_action = action[1]
    wheel3_action = action[2]
    wheel4_action = action[3]
    
    wheel1.publish(str(wheel1_action))
    wheel2.publish(str(wheel2_action))
    wheel3.publish(str(wheel3_action))
    wheel4.publish(str(wheel4_action))
    
    time.sleep(0.5)
 
rate.sleep()