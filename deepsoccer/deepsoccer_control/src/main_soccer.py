#! /usr/bin/env python
############## Basic Import ###############
import math
import numpy as np
import random
import time
import itertools
import os
import sys
from collections import deque
from os import listdir
from os.path import isfile, join, isdir
from pynput import keyboard

############## ROS Import ###############
import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import LaserScan
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

############## ROS Part ###############
reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
pause_simulation = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
unpause_simulation = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

rospy.init_node('jetbot_soccer')
bridge = CvBridge()

'''
class StepWatcher:
    """ A simple class, set to watch its variable. """
    def __init__(self, step_value=0):
        self.step = step_value
        self.value_change_flag = 0

        self.action_list = []
        self.step_list = []
        self.frame_list = []
        self.lidar_list = []

    def set_value(self, new_step_value, image_frame, action, lidar_range):
        self.value_change_flag = 0
        if self.step != new_step_value:
            #print("step value change")
            self.value_change_flag = 1
            self.step = new_step_value
            self.step_list.append(new_step_value)
            self.frame_list.append(image_frame)
            self.action_list.append(action)
            self.lidar_list.append(lidar_range)

            #print("step_watcher.step_list: " + str(step_watcher.step_list))
            #print("step_watcher.action_list: " + str(step_watcher.action_list))
            #print("step_watcher.lidar_list: " + str(step_watcher.lidar_list))
            #self.pre_change()
            #self.post_change()

    def reset(self, step_value=0):
        self.step = step_value
        self.value_change_flag = 0

        self.action_list = []
        self.step_list = []
        self.frame_list = []
        self.lidar_list = []

    #def pre_change(self):
        # do stuff before variable is about to be changed

    #def post_change(self):
        # do stuff right after variable has changed
'''

#step_watcher = StepWatcher()
step_value = 0
action = None
frame = None
lidar_range = None

pathOut = 'jetbot_soccer_data_1.avi'
fps = 5
size = (512,512)
video_out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

key_input = 's'
exit_flag = 0
reset_flag = 0
def on_press(key):
    global key_input
    global action
    #global step_watcher
    global video_out
    global exit_flag

    try:
        key_input = key.char
        #print('alphanumeric key {0} pressed'.format(key.char))

    except AttributeError:
        key_input = key.char
        #print('special key {0} pressed'.format(key))


def on_release(key):
    #print('{0} released'.format(key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False


############## OpenAI Gym Part ###############
def reset():
    #global step_watcher
    reset_simulation()
    #step_watcher.reset()


def step(action_value):
    global action

    action = action_value


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
wheel1 = rospy.Publisher('/robot1/wheel1_velocity_controller/command', Float64, queue_size=5)
wheel2 = rospy.Publisher('/robot1/wheel2_velocity_controller/command', Float64, queue_size=5)
wheel3 = rospy.Publisher('/robot1/wheel3_velocity_controller/command', Float64, queue_size=5)
wheel4 = rospy.Publisher('/robot1/wheel4_velocity_controller/command', Float64, queue_size=5)
roller = rospy.Publisher('/robot1/roller_velocity_controller/command', Float64, queue_size=5)
stick  = rospy.Publisher('/robot1/stick_velocity_controller/command', Float64, queue_size=5)

stop_action = [0, 0, 0, 0, 30, 80]
forward_action = [-30, 30, 30, -30, 30, 0]
left_action = [30, 30, 30, 30, 30, 0]
right_action = [-30, -30, -30, -30, 30, 0]
bacward_action = [30, -30, -30, 30, 30, 0]
hold_action = [0, 0, 0, 0, 30, 0]
kick_action = [0, 0, 0, 0, 0, -80]
robot_action_list = [stop_action, forward_action, left_action, right_action, bacward_action, hold_action, kick_action,]


def image_callback(msg):
    global robot_action_list

    global wheel1
    global wheel2
    global wheel3
    global wheel4
    global roller
    global stick

    #global step_value

    global frame
    global action
    global lidar_range
    # log some info about the image topic
    #rospy.loginfo(msg.header)

    #step_value += 1
    #print("step_value: " + str(step_value))

    # Try to convert the ROS Image message to a CV2 Image
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError, e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    frame = cv2.resize(cv_image, (512, 512), interpolation=cv2.INTER_AREA)
    #action = 1    
    #step_list.append(step.variable)
    #frame_list.append(new_cv_image)
    #action_list.append(action)
    #lidar_list.append(lidar_range)
    #cv2.imshow("Image of Robot Camera", new_cv_image)
    #cv2.waitKey(3)


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


def lidar_callback(msg):
    global lidar_range

    lidar_range = msg.ranges[360]
    if lidar_range == float("inf"):
        lidar_range = 12
    #lidar_list.append(msg.ranges[360])
    #print("len(msg.ranges): " + str(len(msg.ranges)))

    # values at 0 degree
    #print("msg.ranges[0]: " + str(msg.ranges[0]))

    # values at 90 degree
    #print("msg.ranges[360]: " + str(msg.ranges[360]))

    # values at 180 degree
    #print("msg.ranges[719]: " + str(msg.ranges[719]))

    #print("")


sub_image = rospy.Subscriber("/robot1/camera1/image_raw", Image, image_callback)
sub_state = rospy.Subscriber('/gazebo/model_states', ModelStates, state_callback)
sub_lidar = rospy.Subscriber('/jetbot/laser/scan', LaserScan, lidar_callback)

rate = rospy.Rate(2000)

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

action_list = []
step_list = []
frame_list = []
lidar_list = []

pause_simulation()
reset_simulation()

############## ROS + Deep Learning Part ###############
while not rospy.is_shutdown():
    step_value += 1

    #print("rospy loop")
    #print("step_watcher.step: " + str(step_watcher.step))
    #print("step_watcher.pre_step: " + str(step_watcher.pre_step))
    #action_value = random.randint(0,6)
    #step(action_value)
    print("key_input: " + str(key_input))
    if (key_input == 's'):
        action = 0
        unpause_simulation()
    elif (key_input == 'f'):
        action = 1
        unpause_simulation()
    elif (key_input == 'l'):
        action = 2
        unpause_simulation()
    elif (key_input == 'r'):
        action = 3
        unpause_simulation()
    elif (key_input == 'b'):
        action = 4
        unpause_simulation()
    elif (key_input == 'h'):
        action = 5
        unpause_simulation()
    elif (key_input == 'k'):
        action = 6
        unpause_simulation()
    elif (key_input == 'q'):
        print("exit program")

        # Save camera frame as video
        frame_array = np.array(frame_list)
        for i in range(len(frame_array)):
            # writing to a image array
            video_out.write(frame_array[i])

        video_out.release()

        print("step_list: " + str(step_list))
        print("action_list: " + str(action_list))
        print("lidar_list: " + str(lidar_list))

        state = {'step': step_list, 'action': action_list, 'lidar':  lidar_list}
        np.save("jetbot_soccer_data_1.npy", state)

        exit_flag = 1

    #action_value = random.randint(0,6)
    robot_action = robot_action_list[action]
    #print("robot_action: " + str(robot_action))

    wheel1.publish(robot_action[0])
    wheel2.publish(robot_action[1])
    wheel3.publish(robot_action[2])
    wheel4.publish(robot_action[3])
    roller.publish(robot_action[4])
    stick.publish(robot_action[5])

    step_list.append(step_value)
    frame_list.append(frame)
    action_list.append(action)
    lidar_list.append(lidar_range)

    #print("step_value: " + str(step_value))
    #if (step_value == 10):
    #    step_value = 0
    #    reset()

    if exit_flag == 1:
        exit()

    time.sleep(1)
 
rate.sleep()

############## Deep Learning Import ###############
import tensorflow as tf
import per_replay as replay
