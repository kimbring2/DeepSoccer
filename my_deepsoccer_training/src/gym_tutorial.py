#!/usr/bin/env python
import gym
import numpy as np
import time
import math
import random

# ROS packages required
import rospy
import rospkg
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from tf.transformations import euler_from_quaternion, quaternion_from_euler

rospy.init_node('deepsoccer_openai_gym_tutorial', anonymous=True, log_level=rospy.WARN)

task_and_robot_environment_name = rospy.get_param('/deepsoccer_single/task_and_robot_environment_name')
save_path = rospy.get_param('/deepsoccer_single/save_path')
save_file = rospy.get_param('/deepsoccer_single/save_file')

env = StartOpenAI_ROS_Environment('DeepSoccerSingle-v0')

for episode in range(20):
    observation = env.reset()
    for timestep in range(500):
        action = env.action_space.sample()
         
        observation, reward, done, info = env.step(action)
        print("observation camera: " + str(observation[0]))
        print("observation lidar: " + str(observation[1]))
        print("observation infra: " + str(observation[2]))
        print("reward: " + str(reward))
        print("done: " + str(reward))

        if done:
            print("Episode finished after {} timesteps".format(timestep + 1))
            break

env.close()