#!/usr/bin/env python
import gym
import numpy as np
import time
import math
import random
import cv2
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Conv2D, Flatten, TimeDistributed, LSTM, Reshape
import pylab
import os
import glob

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


class OurModel(tf.keras.Model):
    def __init__(self, input_shape, action_space):
        super(OurModel, self).__init__()
        
        self.conv_1 = Conv2D(16, 2, 1, padding="valid", activation="relu", kernel_regularizer='l2')
    	self.conv_2 = Conv2D(32, 3, 1, padding="valid", activation="relu", kernel_regularizer='l2')
    	self.conv_3 = Conv2D(32, 5, 1, padding="valid", activation="relu", kernel_regularizer='l2')

    	self.lstm = LSTM(128, name="core_lstm", return_sequences=True, return_state=True, kernel_regularizer='l2')

        self.dense_0 = Dense(512, activation='relu')
        self.dense_1 = Dense(action_space)
        self.dense_2 = Dense(1)
        
    def call(self, input_, memory_state, carry_state):
    	batch_size = tf.shape(input_)[0]

    	conv_1 = self.conv_1(input_)
    	conv_2 = self.conv_2(conv_1)
    	conv_3 = self.conv_3(conv_2)
    	#print("conv_3.shape: ", conv_3.shape)

        conv_flatten = Flatten()(conv_3)
        conv_reshaped = Reshape((57*57, 32))(conv_flatten)
        #print("conv_reshaped.shape: ", conv_reshaped.shape)

        lstm_outputs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        lstm_output = tf.zeros((57*57, 32))
        for i in range(0, batch_size):
            lstm_input = tf.expand_dims(conv_reshaped[i], 0)
            initial_state = (memory_state, carry_state)
            lstm_output, memory_state, carry_state = self.lstm(lstm_input, initial_state=initial_state)
            lstm_output = tf.squeeze(lstm_output, 0)
            lstm_outputs = lstm_outputs.write(i, lstm_output)

        lstm_outputs = lstm_outputs.stack()
        lstm_outputs_flatten = Flatten()(lstm_outputs)
        #print("lstm_outputs_flatten.shape: ", lstm_outputs_flatten.shape)

        flatten = self.dense_0(lstm_outputs_flatten)
        action_logit = self.dense_1(flatten)
        value = self.dense_2(flatten)
        
        return action_logit, value, memory_state, carry_state


def safe_log(x):
  """Computes a safe logarithm which returns 0 if x is zero."""
  return tf.where(
      tf.math.equal(x, 0),
      tf.zeros_like(x),
      tf.math.log(tf.math.maximum(1e-12, x)))


def take_vector_elements(vectors, indices):
    """
    For a batch of vectors, take a single vector component
    out of each vector.
    Args:
      vectors: a [batch x dims] Tensor.
      indices: an int32 Tensor with `batch` entries.
    Returns:
      A Tensor with `batch` entries, one for each vector.
    """
    return tf.gather_nd(vectors, tf.stack([tf.range(tf.shape(vectors)[0]), indices], axis=1))

mse_loss = tf.keras.losses.MeanSquaredError()
cce_loss = tf.keras.losses.CategoricalCrossentropy()

class A2CAgent:
    # Actor-Critic Main Optimization Algorithm
    def __init__(self, env_name):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        self.env = StartOpenAI_ROS_Environment(env_name)
        self.state_size = (64,64,5)
        self.action_size = self.env.action_space.n
        #print("self.action_size: ", self.action_size)
        self.EPISODES, self.episode, self.max_average_reward, self.min_average_loss = 20000, 0, 10.0, 0.2 # specific for pong
        self.lr = 0.000025

        self.ROWS = 64
        self.COLS = 64
        self.REM_STEP = 4

        # Instantiate plot memory
        self.scores, self.episodes, self.average = [], [], []

        self.Save_Path = 'Models'
        self.workspace_path = '/home/kimbring2/catkin_ws/src/my_deepsoccer_training' 
        
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A2C_{}'.format(self.env_name, self.lr)
        self.model_name = os.path.join(self.Save_Path, self.path)

        # Create Actor-Critic network model
        self.ActorCritic = OurModel(input_shape=self.state_size, action_space=self.action_size)
        
        self.learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def act(self, state, memory_state, carry_state):
        # Use the network to predict the next action to take, using the model
        prediction = self.ActorCritic(state, memory_state, carry_state, training=False)
        action = tf.random.categorical(prediction[0], 1).numpy()

        next_memory_state = prediction[2]
        next_carry_state = prediction[3]

        return action[0][0], next_memory_state, next_carry_state

    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0, len(reward))):
            if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0

            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        #discounted_r -= np.mean(discounted_r) # normalizing the result
        #discounted_r /= np.std(discounted_r) # divide by standard deviation

        return discounted_r
    
    def sl_replay(self, states, actions, memory_state, carry_state):
        # reshape memory to appropriate shape for training
        states = tf.concat(states, 0)
        
        with tf.GradientTape() as tape:
            prediction = self.ActorCritic(states, memory_state, carry_state, training=True)
            action_logits = prediction[0]
            next_memory_state = prediction[2]
            next_carry_state = prediction[3]
            
            actions_onehot = tf.one_hot(actions, self.action_size)
            action_probs = tf.nn.softmax(action_logits)

            action_loss = cce_loss(actions_onehot, action_probs)
            regularization_loss = tf.reduce_sum(self.ActorCritic.losses)

            total_loss = action_loss + 1e-5 * regularization_loss
      
        grads = tape.gradient(total_loss, self.ActorCritic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.ActorCritic.trainable_variables))

        return total_loss, next_memory_state, next_carry_state
        
    def rl_replay(self, states, actions, rewards, memory_state, carry_state):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        
        # Compute discounted rewards
        discounted_r = self.discount_rewards(rewards)
        discounted_r_ = np.vstack(discounted_r)
        with tf.GradientTape() as tape:
            prediction = self.ActorCritic(states, memory_state, carry_state, training=True)
            action_logits = prediction[0]
            values = prediction[1]
            next_memory_state = prediction[2]
            next_carry_state = prediction[3]
            
            action_logits_selected = take_vector_elements(action_logits, actions)
            
            advantages = discounted_r - np.stack(values)[:, 0] 
            
            action_logits_selected = tf.nn.softmax(action_logits_selected)
            action_logits_selected_probs = tf.math.log(action_logits_selected)
            
            actor_loss = -tf.math.reduce_mean(action_logits_selected_probs * advantages) 
            actor_loss = tf.cast(actor_loss, 'float32')
            
            action_probs = tf.nn.softmax(action_logits)
            
            #critic_loss_ = huber_loss(values, discounted_r)
            critic_loss = mse_loss(values, discounted_r_)
            critic_loss = tf.cast(critic_loss, 'float32')
            #print("critic_loss: ", critic_loss)
            total_loss = actor_loss + critic_loss
        
        #print("total_loss: ", total_loss)
        #print("")
            
        grads = tape.gradient(total_loss, self.ActorCritic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.ActorCritic.trainable_variables))
        
        return next_memory_state, next_carry_state

    def load(self, model_name):
        self.ActorCritic = load_model(model_name, compile=False)
        #self.Critic = load_model(Critic_name, compile=False)

    def save(self):
        self.ActorCritic.save(self.model_name)
        #self.Critic.save(self.Model_name + '_Critic.h5')

    pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        if str(episode)[-2:] == "00":# much faster than episode % 100
            pylab.plot(self.episodes, self.scores, 'b')
            pylab.plot(self.episodes, self.average, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.savefig(self.path + ".png")
            except OSError:
                pass

        return self.average[-1]
    
    def sl_train(self):
        file_list = glob.glob(self.workspace_path + "/human_data/*.avi")
        for file in file_list:
        	file_name = file.split('/')[-1].split('.')[0]

        	# Read camera frame data
	        cap = cv2.VideoCapture(self.workspace_path + '/human_data/' + file_name + '.avi')
	        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	        fc = 0
	        ret = True

	        data = np.load(self.workspace_path + '/human_data/' + file_name + ".npy", allow_pickle=True)
	        data_0 = np.reshape(data, 1)
	        data_1 = data_0[0]

	        states, actions = [], []
	        memory_state = tf.zeros([1,128], dtype=np.float32)
	        carry_state = tf.zeros([1,128], dtype=np.float32)
	        while fc < frameCount and ret:
	        	print("fc: ", fc)
	        	ret, image_frame = cap.read()
	        	image_frame_resized = cv2.resize(image_frame, (64,64), interpolation=cv2.INTER_AREA)

	        	frame_state_channel = image_frame_resized / 255.0
	        	lidar_state_channel = (np.ones(shape=(64,64,1), dtype=np.float32)) * data_1['state']['lidar'][fc] / 12
	        	infrared_state_channel = (np.ones(shape=(64,64,1), dtype=np.float32)) * data_1['state']['infrared'][fc] / 2.0

	        	state_channel = np.concatenate((frame_state_channel, lidar_state_channel), axis=2)
	        	state_input = np.concatenate((state_channel, infrared_state_channel), axis=2)
	        	state_input = np.reshape(state_input, (1,64,64,5))

	        	states.append(state_input)
	        	actions.append(data_1['action'][fc])

	        	fc += 1
	        	if fc % 8 == 0:
		            loss, next_memory_state, next_carry_state = self.sl_replay(states, actions, memory_state, carry_state)
		            states, actions = [], []

		            memory_state = next_memory_state
		            carry_state = next_carry_state

		            # Update episode count
		            average_loss = self.PlotModel(loss, self.episode)
		            # saving best models
		            if average_loss <= self.min_average_loss:
		                self.min_average_loss = average_loss
		                #self.save()
		                SAVING = "SAVING"
		            else:
		                SAVING = ""

		            print("episode: {}/{}, average_loss: {:.2f} {}".format(self.episode, self.EPISODES, average_loss, SAVING))
		            if(self.episode < self.EPISODES):
		                self.episode += 1

    def rl_train(self):
        while self.episode < self.EPISODES:
            # Reset episode
            score, done, SAVING = 0, False, ''
            state = self.env.reset()
            frame_state_channel = cv2.resize(state[0], (64,64), interpolation=cv2.INTER_AREA) / 255.0
            lidar_state_channel = (np.ones(shape=(64,64,1), dtype=np.float32)) * state[1] / 12
            infrared_state_channel = (np.ones(shape=(64,64,1), dtype=np.float32)) * state[2] / 2.0

            state_channel = np.concatenate((frame_state_channel, lidar_state_channel), axis=2)
            state_input = np.concatenate((state_channel, infrared_state_channel), axis=2)
            state_input = np.reshape(state_input, (1,64,64,5))

            states, actions, rewards = [], [], []
            memory_state = tf.zeros([1,128], dtype=np.float32)
            carry_state = tf.zeros([1,128], dtype=np.float32)

            initial_memory_state = memory_state
            initial_carry_state = carry_state
            #while not done:
            for i in range(0, 50):
            	print("i: ", i)
                action, next_memory_state, next_carry_state = agent.act(state_input, memory_state, carry_state)
                next_state, reward, done, info = self.env.step(action)
                #print("reward: ", reward)
                frame_next_state_channel = cv2.resize(next_state[0], (64,64), interpolation=cv2.INTER_AREA) / 255.0
                lidar_next_state_channel = (np.ones(shape=(64,64,1), dtype=np.float32)) * next_state[1] / 12
                infrared_next_state_channel = (np.ones(shape=(64,64,1), dtype=np.float32)) * next_state[2] / 2.0

                next_state_channel = np.concatenate((frame_next_state_channel, lidar_next_state_channel), axis=2)
                next_state_input = np.concatenate((next_state_channel, infrared_next_state_channel), axis=2)
                next_state_input = np.reshape(next_state_input, (1,64,64,5))

                states.append(state_input)
                actions.append(action)
                rewards.append(reward)

                score += reward
                state_input = next_state_input

                memory_state = next_memory_state
                carry_state = next_carry_state
                if i % 8 == 0:    
                    next_memory_state, next_carry_state = self.rl_replay(states, actions, rewards, 
                                                                                     initial_memory_state, initial_carry_state)
                    states, actions, rewards = [], [], []
                    
                    memory_state = next_memory_state
                    carry_state = next_carry_state

            # Update episode count
            average_reward = self.PlotModel(score, self.episode)
            # saving best models
            if average_reward >= self.max_average_reward:
                self.max_average_reward = average_reward
                #self.save()
                SAVING = "SAVING"
            else:
                SAVING = ""

            print("episode: {}/{}, score: {}, average_reward: {:.2f} {}".format(self.episode, self.EPISODES, score, average_reward, SAVING))
            if(self.episode < self.EPISODES):
                self.episode += 1

        env.close()            

    def test(self, Actor_name, Critic_name):
        self.load(Actor_name, Critic_name)
        for e in range(100):
            state = self.reset(self.env)
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax(self.Actor.predict(state))
                state, reward, done, _ = self.step(action, self.env, state)
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break

        self.env.close()


env_name = 'DeepSoccerSingle-v0'
agent = A2CAgent(env_name)

agent.rl_train()
#agent.sl_train()
'''
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
'''