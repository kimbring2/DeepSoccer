#!/usr/bin/env python
import gym
import numpy as np
import time
import yaml
import math
import random

# RL packages required
import tensorflow as tf
import cv2

from ForgER.agent import *
from ForgER.replay_buff import AggregatedBuff
from ForgER.model import get_network_builder

#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_virtual_device_configuration(gpus[0],
#            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

with open('/home/kimbring2/catkin_ws/src/my_deepsoccer_training/src/deepsoccer_config.yaml', "r") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

agent_config = config['agent']
buffer_config = config['buffer']

env_dict = {'to_demo': 'float32', 'demo': 'float32', 'n_done': 'bool', 'actual_n': 'float32', 'n_reward': 'float32',
              'done': 'bool', 'indexes': 'int32', 'next_state': 'float32', 'state': 'float32', 'weights': 'float32', 'n_state': 'float32', 
              'action': 'int32', 'reward': 'float32'}
dtype_dict = {'to_demo': 'float32', 'demo': 'float32', 'n_done': 'bool', 'actual_n': 'float32', 'n_reward': 'float32',
                'done': 'bool', 'indexes': 'int32', 'next_state': 'float32', 'state': 'float32', 'weights': 'float32', 'n_state': 'float32', 
                'action': 'int32', 'reward': 'float32'}
replay_buffer = AggregatedBuff(env_dict=env_dict, 
                                      size=config['buffer']['size'],
                                      episodes_to_decay=config['buffer']['episodes_to_decay'], 
                                      min_demo_proportion=config['buffer']['min_demo_proportion'])
make_model = get_network_builder('deepsoccer_dqfd')

#print("env.observation_space: " + str(env.observation_space))
#print("env.action_space: " + str(env.action_space))

#env.observation_space: Box(144,)
#env.action_space: Discrete(7)

obs_space = (64, 64, 3)
action_space = 8

agent = Agent(agent_config, config['buffer']['size'], replay_buffer, make_model, obs_space, action_space, dtype_dict)
#agent.load("/home/kimbring2/catkin_ws/src/my_deepsoccer_training/src/train/deepsoccer_single/pre_trained_model/variables/variables")
agent.add_demo()
agent.train(config['pretrain']['steps'])
#agent.test(env)
env.close()