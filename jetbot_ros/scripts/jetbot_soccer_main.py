#!/usr/bin/env python3
############## ROS Import ###############
import rospy
import std_msgs
from sensor_msgs.msg import Image

import numpy as np
import random
import time
import itertools
import os
import cv2
from cv_bridge import CvBridge, CvBridgeError

#import jetson.inference
#import jetson.utils

#net = jetson.inference.detectNet("ssd-mobilenet-v2")


import tensorflow as tf

imported = tf.saved_model.load("/home/kimbring2/Desktop/pre_trained_model.ckpt")
f = imported.signatures["serving_default"]
test_input = np.zeros([1,128,128,5])

test_tensor = tf.convert_to_tensor(test_input, dtype=tf.float32)
#print(f(test_tensor)['dueling_model'].numpy()[0])

bridge = CvBridge()

camera_frame = np.zeros([128,128,3])
def image_callback(msg):
    global camera_frame
    
    #print("image_callback")
    cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
    #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2RGBA).astype(np.float32)
    camera_frame = cv2.resize(cv_image, (128, 128), interpolation=cv2.INTER_AREA)
    #print("cv_image.shape: " + str(cv_image.shape))
    #print("type(cv_image): " + str(type(cv_image)))
    
    #width = 1280
    #height = 720
    #cuda_image = jetson.utils.cudaFromNumpy(cv_image)
    #array = jetson.utils.cudaToNumpy(cuda_image, 720, 1280, 3)
    #jetson.utils.saveImageRGBA("cuda-from-numpy.jpg", cuda_image, 720, 1280)
    #print("cv_image.shape: " + str(cv_image.shape))
    #detections = net.Detect(cuda_image, width, height, "box,labels,conf")

    # print the detections
    #print("detected {:d} objects in image".format(len(detections)))

    #for detection in detections:
    #    print(detection)
    
    cv2.imwrite("camera_frame.jpg", camera_frame)
    #jetson.utils.saveImageRGBA("detected_image.jpg", cv_image, width, height)


lidar_value = 0
def lidar_callback(msg):
    global lidar_value
    lidar_value = msg.data
    
    #print("lidar: " + str(msg))

    
infrared_value = 'False'
def infrared_callback(msg):
    global infrared_value
    
    infrared_value = msg.data
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

stop_action = [0, 0, 0, 0, 'stop', 'none']
forward_action = [50, 1074, 1074, 50, 'in', 'none']
left_action = [1074, 1074, 1074, 1074, 'in', 'none']
right_action = [50, 50, 50, 50, 'in', 'out']
bacward_action = [1074, 50, 50, 1074, 'in', 'none']
hold_action = [0, 0, 0, 0, 'in', 'none']
kick_action = [0, 0, 0, 0, 'stop', 'out']
run_action = [100, 1124, 1124, 100, 'stop', 'out']
robot_action_list = [stop_action, forward_action, left_action, right_action, bacward_action, hold_action, kick_action, run_action]

############## ROS + Deep Learning Part ###############
while not rospy.is_shutdown():
    #action_index = 0
    #print("camera_frame.shape: " + str(camera_frame.shape))
    #print("lidar_value: " + str(lidar_value))
    lidar_ = int(lidar_value) / 1200
    #print("lidar_: " + str(lidar_))
    
    #print("infrared_value: " + str(infrared_value))
    #print("type(infrared_value): " + str(type(infrared_value)))
    infrared_ = int(infrared_value == 'True')
    #print("infrared_: " + str(infrared_))
    #print("action: " + str(action))
    #print("")
    
    frame_state_channel = camera_frame / 255.0
    lidar_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * lidar_
    infrared_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * infrared_ / 2.0
    state_channel1 = np.concatenate((frame_state_channel, lidar_state_channel), axis=2)
    state_channel2 = np.concatenate((state_channel1, infrared_state_channel), axis=2)
    #print("state_channel2.shape: " + str(state_channel2.shape))
    
    state_channel_tensor = tf.convert_to_tensor(state_channel2, dtype=tf.float32)
    predict_value = f(test_tensor)['dueling_model'].numpy()[0]
    print("predict_value: " + str(predict_value))
    action_index = np.argmax(an_predict_valuearray, axis=0)
    print("action_index: " + str(action_index))
    action = robot_action_list[action_index]
    
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