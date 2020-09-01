#!/usr/bin/env python
import gym
import numpy as np
import time
import cv2
from pynput import keyboard

# ROS packages required
import rospy
import rospkg
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

rospy.init_node('example_deepsoccer_soccer_qlearn', anonymous=True, log_level=rospy.WARN)

task_and_robot_environment_name = rospy.get_param('/deepsoccer/task_and_robot_environment_name')
save_path = rospy.get_param('/deepsoccer/save_path')
save_file = rospy.get_param('/deepsoccer/save_file')

env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
key_input = 'f'


def on_press(key):
    global key_input
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


def reset_pose():
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
    state_model.model_name = "robot1"
    state_model.pose = pose
    resp = set_state(state_model)


path_video = save_path + save_file + '.avi'
fps = 5
size = (512,512)
video_out = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

frame_list = []
cur_lidar_list = []
cur_infrared_list = []
next_lidar_list = []
next_infrared_list = []

episode_list = []
action_list = []
reward_list = []
done_list = []
step_list = []
state_list = []
next_state_list = []

for i_episode in range(20):
    observation = env.reset()
    reset_pose()
    for t in range(5000):
        #env.render()
        #print("observation[0].shape: " + str(observation[0].shape))

        #cv2.imshow("obs_image", obs_image)
        #cv2.waitKey(3)

        print("lidar: " + str(observation[1]))
        print("infrared: " + str(observation[2]))
        #print("infra: " + str(observation[2]))
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

            save_data = {'episode': i_episode, 'step': step_list, 
                          'state': {'lidar': cur_lidar_list, 'infrared': cur_infrared_list}, 
                          'next_state': {'lidar': next_lidar_list, 'infrared': next_infrared_list}, 
                          'action': action_list, 'reward': reward_list, 'done': done_list}

            path_npy = save_path + save_file + '.npy'
            np.save(path_npy, save_data)

            #print("episode_list: " + str(episode_list))
            #print("step_list: " + str(step_list))
            #print("state_list: " + str(state_list))
            #print("state_list: " + str(state_list))
            #print("action_list: " + str(action_list))
            #print("reward_list: " + str(reward_list))
            #print("done_list: " + str(done_list))

            exit()

        observation1, reward, done, info = env.step(action)
        episode_list.append(i_episode)
        step_list.append(t)

        frame_list.append(observation[0])

        #state = {'lidar': observation[1], 'infrared': observation[2]}
        cur_lidar_list.append(observation[1])
        cur_infrared_list.append(observation[2])
        #state_list.append(state)

        #next_state = {'lidar': observation1[1], 'infrared': observation1[2]}
        next_lidar_list.append(observation1[1])
        next_infrared_list.append(observation1[2])
        action_list.append(action)
        reward_list.append(reward)
        done_list.append(done)

        observation = observation1

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

env.close()