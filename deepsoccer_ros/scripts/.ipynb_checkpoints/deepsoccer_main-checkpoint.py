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

path_raw_video = '/home/kimbring2/Desktop/raw_video.avi'
path_styled_video = '/home/kimbring2/Desktop/styled_video.avi'
fps = 5
size = (512,512)

raw_video_out = cv2.VideoWriter(path_raw_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (1280,720))
styled_video_out = cv2.VideoWriter(path_styled_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (128,128))

imported_rl = tf.saved_model.load("/home/kimbring2/Desktop/rl_model")
imported_style = tf.saved_model.load("/home/kimbring2/Desktop/style_model")
#imported_cyclegan = tf.saved_model.load("/home/kimbring2/Desktop/cyclegan_model")

f_rl = imported_rl.signatures["serving_default"]
f_style = imported_style.signatures["serving_default"]
#f_cyclegan = imported_cyclegan.signatures["serving_default"]

rl_test_input = np.zeros([1,128,128,5])
style_test_input = np.zeros([1,256,256,3])
#cyclegan_test_input = np.zeros([1,256,256,3])

rl_test_tensor = tf.convert_to_tensor(rl_test_input, dtype=tf.float32)
style_test_tensor = tf.convert_to_tensor(style_test_input, dtype=tf.float32)
#cyclegan_test_tensor = tf.convert_to_tensor(cyclegan_test_input, dtype=tf.float32)

print(f_rl(rl_test_tensor)['dueling_model'].numpy()[0])
time.sleep(2)
print(f_style(style_test_tensor)['output_1'].numpy()[0])
#print(f_cyclegan(cyclegan_test_tensor)['output_1'].numpy()[0])

bridge = CvBridge()

camera_frame = np.zeros([128,128,3])
def image_callback(msg):
    global camera_frame
    
    #print("image_callback")
    cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
    #raw_video_out.write(cv_image)
        
    cv_image_shape = cv_image.shape
    #print("cv_image.shape: " + str(cv_image.shape))
    width = cv_image_shape[1]
    height = cv_image_shape[0]
    
    cv_image = cv2.resize(cv_image, (256, 256), interpolation=cv2.INTER_AREA)
    cv_image = cv2.normalize(cv_image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2RGBA).astype(np.float32)
    
    resized = np.array([cv_image])
    input_tensor = tf.convert_to_tensor(resized, dtype=tf.float32)
    output = f_style(input_tensor)['output_1'].numpy()[0]
    
    #camera_frame = cv2.resize(cv_image, (128, 128), interpolation=cv2.INTER_AREA)
    camera_frame = cv2.resize(output, (128, 128), interpolation=cv2.INTER_AREA)
    #print("camera_frame.shape: " + str(camera_frame.shape))
    #print("")
    #styled_video_out.write(np.uint8(camera_frame))
    #print("cv_image.shape: " + str(cv_image.shape))
    #print("type(cv_image): " + str(type(cv_image)))
    
    #cuda_image = jetson.utils.cudaFromNumpy(cv_image)
    #array = jetson.utils.cudaToNumpy(cuda_image, 720, 1280, 3)
    #jetson.utils.saveImageRGBA("cuda-from-numpy.jpg", cuda_image, 720, 1280)
    #print("cv_image.shape: " + str(cv_image.shape))
    #detections = net.Detect(cuda_image, width, height, "box,labels,conf")

    # print the detections
    #print("detected {:d} objects in image".format(len(detections)))

    #for detection in detections:
    #    print(detection)
    
    #cv2.imwrite("camera_frame.jpg", camera_frame)
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
wheel1 = rospy.Publisher('/deepsoccer_motors/cmd_str_wheel1', std_msgs.msg.String, queue_size=1)
wheel2 = rospy.Publisher('/deepsoccer_motors/cmd_str_wheel2', std_msgs.msg.String, queue_size=1)
wheel3 = rospy.Publisher('/deepsoccer_motors/cmd_str_wheel3', std_msgs.msg.String, queue_size=1)
wheel4 = rospy.Publisher('/deepsoccer_motors/cmd_str_wheel4', std_msgs.msg.String, queue_size=1)
solenoid = rospy.Publisher('/deepsoccer_solenoid/cmd_str', std_msgs.msg.String, queue_size=5)
roller = rospy.Publisher('/deepsoccer_roller/cmd_str', std_msgs.msg.String, queue_size=5)
rospy.Subscriber("/deepsoccer_camera/raw", Image, image_callback)
rospy.Subscriber("/deepsoccer_lidar", std_msgs.msg.String, lidar_callback)
rospy.Subscriber("/deepsoccer_infrared", std_msgs.msg.String, infrared_callback)
 
rate = rospy.Rate(5000)

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
    #print("start")
    
    action_index = 0
    #print("camera_frame.shape: " + str(camera_frame.shape))
    #print("lidar_value: " + str(lidar_value))
    lidar_ = int(lidar_value) / 1200
    print("lidar_: " + str(lidar_))
    
    #print("infrared_value: " + str(infrared_value))
    #print("type(infrared_value): " + str(type(infrared_value)))
    infrared_ = int(infrared_value == 'True')
    print("infrared_: " + str(infrared_))
    #print("action: " + str(action))
    #print("")
    
    frame_state_channel = camera_frame
    lidar_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * lidar_
    infrared_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * infrared_ / 2.0
    state_channel1 = np.concatenate((frame_state_channel, lidar_state_channel), axis=2)
    state_channel2 = np.concatenate((state_channel1, infrared_state_channel), axis=2)
    state_channel2 = np.array([state_channel2])
    #print("state_channel2.shape: " + str(state_channel2.shape))
    
    state_channel_tensor = tf.convert_to_tensor(state_channel2, dtype=tf.float32)
    predict_value = f_rl(state_channel_tensor)['dueling_model'].numpy()[0]
    #print("predict_value: " + str(predict_value))
    action_index = np.argmax(predict_value, axis=0)
    #action_index = 0
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
    
    #time.sleep(0.1)
 
rate.sleep()