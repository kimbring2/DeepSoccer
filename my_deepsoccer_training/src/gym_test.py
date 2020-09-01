```
#!/usr/bin/env python
import gym
import numpy
import time
import cv2

# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

rospy.init_node('example_deepsoccer_soccer_qlearn', anonymous=True, log_level=rospy.WARN)

task_and_robot_environment_name = rospy.get_param('/deepsoccer/task_and_robot_environment_name')
env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)

for i_episode in range(20):
    observation = env.reset()

    for t in range(100):
        #env.render()
        #print("observation[0].shape: " + str(observation[0].shape))

        obs_image = observation[0]
        cv2.imshow("obs_image", obs_image)
        cv2.waitKey(3)

        print("observation[1]: " + str(observation[1]))
        print("observation[2]: " + str(observation[2]))
        print("observation[3]: " + str(observation[3]))
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
```
