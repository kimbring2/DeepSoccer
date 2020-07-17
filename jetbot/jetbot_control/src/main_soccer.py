#! /usr/bin/env python
import math
import numpy as np
import random
import time
import itertools
import os
from collections import deque
from os import listdir
from os.path import isfile, join, isdir

############## ROS Import ###############
import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose

############## Image Import ###############
from cv_bridge import CvBridge, CvBridgeError
import cv2

############## Deep Learning Import ###############
import tensorflow as tf
import per_replay as replay

############## ROS Part ###############
reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
rospy.init_node('jetbot_soccer')
bridge = CvBridge()


def image_callback(msg):
    # log some info about the image topic
    #rospy.loginfo(msg.header)

    # Try to convert the ROS Image message to a CV2 Image
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError, e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    new_cv_image = cv2.resize(cv_image, (512, 512), interpolation=cv2.INTER_AREA)

    cv2.imshow("Image of Robot Camera", new_cv_image)
    cv2.waitKey(3)


def state_callback(msg):
    #print("msg.name: " + str(msg.name))
    # msg.name: ['ground_plane', 'field', 'left_goal', 'right_goal', 'football', 'jetbot']

    robot1_index = (msg.name).index("robot1")
    football_index = (msg.name).index("football")
    left_goal_index = (msg.name).index("left_goal")
    right_goal_index = (msg.name).index("right_goal")
    #print("football_index: " + str(football_index))

    robot1_pose = (msg.pose)[robot1_index]
    #print("robot1_pose.position: " + str(robot1_pose.position))

    football_pose = (msg.pose)[football_index]
    #print("football_pose.position: " + str(football_pose.position))

    left_goal_pose = (msg.pose)[left_goal_index]
    #print("left_goal_pose.position: " + str(left_goal_pose.position))

    right_goal_pose = (msg.pose)[right_goal_index]
    #print("right_goal_pose.position: " + str(right_goal_pose.position))

    #print("football_pose.position.x: " + str(football_pose.position.x))
    #print("left_goal_pose.position.x: " + str(left_goal_pose.position.x))
    #print("left_goal_pose.position.y: " + str(left_goal_pose.position.y))
    if (football_pose.position.x < left_goal_pose.position.x):
        d = True
    elif (football_pose.position.x > right_goal_pose.position.x):
        d = True

pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
wheel1 = rospy.Publisher('/robot1/wheel1_velocity_controller/command', Float64, queue_size=5)
wheel2 = rospy.Publisher('/robot1/wheel2_velocity_controller/command', Float64, queue_size=5)
wheel3 = rospy.Publisher('/robot1/wheel3_velocity_controller/command', Float64, queue_size=5)
wheel4 = rospy.Publisher('/robot1/wheel4_velocity_controller/command', Float64, queue_size=5)
roller = rospy.Publisher('/robot1/roller_velocity_controller/command', Float64, queue_size=5)
stick  = rospy.Publisher('/robot1/stick_velocity_controller/command', Float64, queue_size=5)

sub_image = rospy.Subscriber("/robot1/camera1/image_raw", Image, image_callback)
sub_state = rospy.Subscriber('/gazebo/model_states', ModelStates, state_callback)

rate = rospy.Rate(2000)

stop_action = [0, 0, 0, 0, 30, 80]
forward_action = [-30, 30, 30, -30, 30, 0]
left_action = [30, 30, 30, 30, 30, 0]
right_action = [-30, -30, -30, -30, 30, 0]
bacward_action = [30, -30, -30, 30, 30, 0]
hold_action = [0, 0, 0, 0, 30, 0]
kick_action = [0, 0, 0, 0, 0, -80]
robot_action_list = [stop_action, forward_action, left_action, right_action, bacward_action, hold_action, kick_action,]

set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
pose = Pose() 
pose.position.x = np.random.randint(1,20) / 10.0
pose.position.y = np.random.randint(1,20) / 10.0
pose.position.z = 0.12
  
pose.orientation.x = 0
pose.orientation.y = 0
pose.orientation.z = 0
pose.orientation.w = 0
    
state_model = ModelState()   
state_model.model_name = "football"
state_model.pose = pose
resp = set_state(state_model)


############## ROS + Deep Learning Part ###############
while not rospy.is_shutdown():
    key_input = raw_input('Enter your input:')
    print("key_input: " + str(key_input))
    
    a = None
    if (key_input == 's'):
        a = 0
    elif (key_input == 'f'):
        a = 1
    elif (key_input == 'l'):
        a = 2
    elif (key_input == 'r'):
        a = 3
    elif (key_input == 'b'):
        a = 4
    elif (key_input == 'h'):
        a = 5
    elif (key_input == 'k'):
        a = 6

    robot_action = robot_action_list[a]

    wheel1.publish(robot_action[0])
    wheel2.publish(robot_action[1])
    wheel3.publish(robot_action[2])
    wheel4.publish(robot_action[3])
    roller.publish(robot_action[4])
    stick.publish(robot_action[5])
    
    time.sleep(0.5)
 
rate.sleep()
