#! /usr/bin/env python
############## ROS Import ###############
import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge, CvBridgeError
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import math
from math import atan2
import numpy as np
import random
import time
import itertools
import os


# maybe do some 'wait for service' here
reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

############## ROS Part ###############
rospy.init_node('jetbot')

pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

wheel1 = rospy.Publisher('/robot1/wheel1_velocity_controller/command', Float64, queue_size=5)
wheel2 = rospy.Publisher('/robot1/wheel2_velocity_controller/command', Float64, queue_size=5)
wheel3 = rospy.Publisher('/robot1/wheel3_velocity_controller/command', Float64, queue_size=5)
wheel4 = rospy.Publisher('/robot1/wheel4_velocity_controller/command', Float64, queue_size=5)
roller = rospy.Publisher('/robot1/roller_velocity_controller/command', Float64, queue_size=5)
stick  = rospy.Publisher('/robot1/stick_velocity_controller/command', Float64, queue_size=5)
 
rate = rospy.Rate(2000)

stop_action = [0, 0, 0, 0, 30, 80]
forward_action = [-30, 30, 30, -30, 30, 0]
left_action = [30, 30, 30, 30, 30, 0]
right_action = [-30, -30, -30, -10, 30, 0]
bacward_action = [30, -30, -30, 10, 30, 0]
hold_action = [0, 0, 0, 0, 30, 0]
kick_action = [0, 0, 0, 0, 0, -80]
robot_action_list = [stop_action, forward_action, left_action, right_action, bacward_action, hold_action, kick_action,]

set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
pose = Pose() 
pose.position.x = np.random.randint(1,20) / 10.0
pose.position.y = np.random.randint(1,20) / 10.0
#pose.position.x = 0
#pose.position.y = 0
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
