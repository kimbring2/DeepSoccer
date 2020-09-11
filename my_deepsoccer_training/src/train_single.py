#!/usr/bin/env python
import gym
import numpy as np
import time
import yaml

# ROS packages required
import rospy
import rospkg
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

# RL packages required
import tensorflow as tf
import cv2

from ForgER.agent import *
from ForgER.replay_buff import AggregatedBuff
from ForgER.model import get_network_builder

rospy.init_node('example_deepsoccer_soccer_qlearn', anonymous=True, log_level=rospy.WARN)

task_and_robot_environment_name = rospy.get_param('/deepsoccer/task_and_robot_environment_name')
save_path = rospy.get_param('/deepsoccer/save_path')
save_file = rospy.get_param('/deepsoccer/save_file')

env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)

def get_dtype_dict(env):
    action_shape = env.action_space.shape
    action_shape = action_shape if len(action_shape) > 0 else 1
    action_dtype = env.action_space.dtype
    action_dtype = 'int32' if np.issubdtype(action_dtype, int) else action_dtype
    action_dtype = 'float32' if np.issubdtype(action_dtype, float) else action_dtype
    env_dict = {'action': {'shape': action_shape,
                            'dtype': action_dtype},
                'reward': {'dtype': 'float32'},
                'done': {'dtype': 'bool'},
                'n_reward': {'dtype': 'float32'},
                'n_done': {'dtype': 'bool'},
                'actual_n': {'dtype': 'float32'},
                'demo': {'dtype': 'float32'},
                'to_demo': {'dtype': 'float32'},
                'state': {'dtype': 'float32'},
                'next_state': {'dtype': 'float32'},
                'n_state': {'dtype': 'float32'}
                }
    for prefix in ('', 'next_', 'n_'):
        if isinstance(env.observation_space, gym.spaces.Dict):
            for name, space in env.observation_space.spaces.items():
                env_dict[prefix + name] = {'shape': space.shape,
                                           'dtype': space.dtype}
        else:
            env_dict[prefix + 'state'] = {'shape': env.observation_space.shape,
                                          'dtype': env.observation_space.dtype}
    dtype_dict = {key: value['dtype'] for key, value in env_dict.items()}
    dtype_dict.update(weights='float32', indexes='int32')
    return env_dict, dtype_dict


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


with open('/home/kimbring2/catkin_ws/src/my_deepsoccer_training/src/deepsoccer_config.yaml', "r") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

agent_config = config['agent']
buffer_config = config['buffer']
#print("config['buffer']['episodes_to_decay']: " + str(config['buffer']['episodes_to_decay']))
#print("config['buffer']['min_demo_proportion']: " + str(config['buffer']['min_demo_proportion']))

_, dtype_dict = get_dtype_dict(env)
print("dtype_dict: " + str(dtype_dict))
env_dict = {'to_demo': 'float32', 'demo': 'float32', 'n_done': 'bool', 'actual_n': 'float32', 'n_reward': 'float32',
             'done': 'bool', 'indexes': 'int32', 'next_state': 'float32', 'state': 'float32', 'weights': 'float32', 'n_state': 'float32', 
             'action': 'int32', 'reward': 'float32'}
dtype_dict = {'to_demo': 'float32', 'demo': 'float32', 'n_done': 'bool', 'actual_n': 'float32', 'n_reward': 'float32',
             'done': 'bool', 'indexes': 'int32', 'next_state': 'float32', 'state': 'float32', 'weights': 'float32', 'n_state': 'float32', 
             'action': 'int32', 'reward': 'float32'}
#print("dtype_dict: " + str(dtype_dict))
replay_buffer = AggregatedBuff(env_dict=env_dict, 
                                  size=config['buffer']['size'],
                                  episodes_to_decay=config['buffer']['episodes_to_decay'], 
                                  min_demo_proportion=config['buffer']['min_demo_proportion'])
make_model = get_network_builder('deepsoccer_dqfd')


#print("env.observation_space: " + str(env.observation_space))
#print("env.action_space: " + str(env.action_space))

#env.observation_space: Box(144,)
#env.action_space: Discrete(7)

agent = Agent(agent_config, replay_buffer, make_model, env.observation_space,
                 env.action_space, dtype_dict)

#agent.add_demo()
#env.reset()
#agent.pre_train(config['pretrain']['steps'])
#scores_, _ = agent.train(env, name="model.ckpt", episodes=config['episodes'])
#print("scores_: " + str(scores_))
agent.test(env)
env.close()