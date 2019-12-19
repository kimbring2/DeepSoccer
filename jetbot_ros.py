#!/usr/bin/env python2.7
# Import ROS libraries and messages
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import time
import numpy as np

# Import Jetbot motor libraries
from Adafruit_MotorHAT import Adafruit_MotorHAT

# Import OpenCV libraries and tools
import cv2
from cv_bridge import CvBridge, CvBridgeError

# Import Jetson Inference libraries and tools
import jetson.inference
import jetson.utils

import argparse

#import tensorrt as trt

# Print "Hello Jetbot!" to terminal
print "Hello Jetbot!"

# Initialize the ROS Node named 'opencv_example', allow multiple nodes to be run with this name
rospy.init_node('Jetbot_ROS', anonymous=True)

# Print "Hello ROS!" to the Terminal and to a ROS Log file located in ~/.ros/log/loghash/*.log
rospy.loginfo("Hello Jetbot!")

# Initialize the CvBridge class
bridge = CvBridge()

# load the recognition network
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.3)

# Define a callback for the Image message
def image_callback(img_msg):
    # log some info about the image topic
    rospy.loginfo(img_msg.header)

    # Try to convert the ROS Image message to a CV2 Image
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError, e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    # Flip the image 360deg
    cv_image = cv2.flip(cv_image, 0)
    
    b_channel, g_channel, r_channel = cv2.split(cv_image)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 # creating a dummy alpha channel image.
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    in_arr = jetson.utils.cudaFromNumpy(img_BGRA)
    
    #jetson.utils.saveImageRGBA('ros_camera_image.jpg', in_arr, 1280, 720)
    overlay = "box,labels,conf"
    
    width = 1280
    height = 720
    
    # classify the image
    detections = net.Detect(in_arr, width, height, overlay)
    
    # print the detections
    print("detected {:d} objects in image".format(len(detections)))
    
    # find the object description
    jetson.utils.saveImageRGBA("detect_result.jpg", in_arr, width, height)

# Initalize a subscriber to the "/jetbot_camera/raw" topic with the function "image_callback" as a callback
sub_image = rospy.Subscriber("/jetbot_camera/raw", Image, image_callback)

# Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
while not rospy.is_shutdown():
    motor_driver = Adafruit_MotorHAT(i2c_bus=1)

    motor_left_ID = 1
    motor_right_ID = 2

    motor_left = motor_driver.getMotor(motor_left_ID)
    motor_right = motor_driver.getMotor(motor_right_ID)
    
    motor_left.setSpeed(0)
    motor_right.setSpeed(0)
    
    motor_left.run(Adafruit_MotorHAT.FORWARD)
    motor_right.run(Adafruit_MotorHAT.FORWARD)
    
    rospy.spin()
