#!/usr/bin/env python
import gym
import numpy as np
import time
import cv2
from pynput import keyboard

# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

rospy.init_node('example_deepsoccer_soccer_qlearn', anonymous=True, log_level=rospy.WARN)

task_and_robot_environment_name = rospy.get_param('/deepsoccer/task_and_robot_environment_name')
env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)

pathOut = '/home/kimbring2/jetbot_soccer_data_1.avi'
fps = 5
size = (512,512)
video_out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i_episode in range(20):
    observation = env.reset()

    frame_list = []
    action_list = []
    step_list = []
    lidar_list = []
    for t in range(5000):
        #env.render()
        #print("observation[0].shape: " + str(observation[0].shape))

        #cv2.imshow("obs_image", obs_image)
        #cv2.waitKey(3)

        #print("observation[1]: " + str(observation[1]))
        #print("observation[2]: " + str(observation[2]))
        action = env.action_space.sample()
        
        print("key_input: " + str(key_input))
        if (key_input == 's'):
            action = 0
        elif (key_input == 'f'):
            action = 1
        elif (key_input == 'l'):
            action = 2
        elif (key_input == 'r'):
            action = 3
        elif (key_input == 'b'):
            action = 4
        elif (key_input == 'h'):
            action = 5
        elif (key_input == 'k'):
            action = 6
        elif (key_input == 'p'):
            action = 7
        elif (key_input == 'q'):
            print("exit program")

            # Save camera frame as video
            frame_array = np.array(frame_list)
            for i in range(len(frame_array)):
                # writing to a image array
                video_out.write(frame_array[i])

            video_out.release()

            #print("step_list: " + str(step_list))
            #print("action_list: " + str(action_list))
            #print("lidar_list: " + str(lidar_list))

            state = {'step': step_list, 'action': action_list, 'lidar':  lidar_list}
            np.save("jetbot_soccer_data_1.npy", state)

            exit()

        observation, reward, done, info = env.step(action)
        step_list.append(t)
        #frame_list.append(observation[0])
        action_list.append(action)
        #lidar_list.append(observation[1])

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()