#!/usr/bin/env python
import gym
import numpy as np
import time
import math
import random
import os
import glob

import cv2
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Conv2D, Flatten, TimeDistributed, LSTM, Reshape

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
gpu_use = rospy.get_param('/deepsoccer_single/gpu_use')
workspace_path = rospy.get_param('/deepsoccer_single/workspace_path')
sl_training = rospy.get_param('/deepsoccer_single/sl_training')
learning_rate = rospy.get_param('/deepsoccer_single/learning_rate')
save_model = rospy.get_param('/deepsoccer_single/save_model')
pretrained_model = rospy.get_param('/deepsoccer_single/pretrained_model')
testing = rospy.get_param('/deepsoccer_single/testing')

if gpu_use == True:
    os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class DeepSoccerModel(tf.keras.Model):
    def __init__(self, action_space):
        super(DeepSoccerModel, self).__init__()
        
        self.conv_1 = Conv2D(16, 8, 4, padding="valid", activation="relu", kernel_regularizer='l2')
    	self.conv_2 = Conv2D(32, 4, 2, padding="valid", activation="relu", kernel_regularizer='l2')
    	self.conv_3 = Conv2D(32, 3, 1, padding="valid", activation="relu", kernel_regularizer='l2')

    	self.lstm = LSTM(128, name="core_lstm", return_sequences=True, return_state=True, kernel_regularizer='l2')

        self.dense_0 = Dense(512, activation='relu', kernel_regularizer='l2')
        self.dense_1 = Dense(action_space, kernel_regularizer='l2')
        self.dense_2 = Dense(1, kernel_regularizer='l2')
        
    def call(self, input_, memory_state, carry_state):
    	batch_size = tf.shape(input_)[0]

    	conv_1 = self.conv_1(input_)
    	conv_2 = self.conv_2(conv_1)
    	conv_3 = self.conv_3(conv_2)
    	#print("conv_3.shape: ", conv_3.shape)

        conv_flatten = Flatten()(conv_3)
        conv_reshaped = Reshape((12*12, 32))(conv_flatten)
        #print("conv_reshaped.shape: ", conv_reshaped.shape)

        #lstm_outputs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        #lstm_output = tf.zeros((12*12, 32))
        #for i in range(0, batch_size):
        #lstm_input = tf.expand_dims(conv_reshaped[i], 0)
        initial_state = (memory_state, carry_state)
        lstm_output, memory_state, carry_state = self.lstm(conv_reshaped, initial_state=initial_state)
        #lstm_output = tf.squeeze(lstm_output, 0)
        #lstm_outputs = lstm_outputs.write(i, lstm_output)

        #lstm_outputs = lstm_outputs.stack()
        lstm_output_flatten = Flatten()(lstm_output)
        #print("lstm_outputs_flatten.shape: ", lstm_outputs_flatten.shape)

        flatten = self.dense_0(lstm_output_flatten)
        action_logit = self.dense_1(flatten)
        value = self.dense_2(flatten)
        
        return action_logit, value, memory_state, carry_state


def get_model(action_space):
    input_ = tf.keras.Input(shape=(128, 128, 5))
    memory_state = tf.keras.Input(shape=(128)) 
    carry_state = tf.keras.Input(shape=(128))

    conv_1 = Conv2D(16, 8, 4, padding="valid", activation="relu", kernel_regularizer='l2')(input_)
    conv_2 = Conv2D(32, 4, 2, padding="valid", activation="relu", kernel_regularizer='l2')(conv_1)
    conv_3 = Conv2D(32, 3, 1, padding="valid", activation="relu", kernel_regularizer='l2')(conv_2)

    conv_flatten = Flatten()(conv_3)
    conv_reshaped = Reshape((12*12, 32))(conv_flatten)

    initial_state = (memory_state, carry_state)
    lstm_output, final_memory_state, final_carry_state = LSTM(128, name="core_lstm", return_sequences=True, return_state=True, 
        kernel_regularizer='l2')(conv_reshaped, initial_state=initial_state)

    lstm_output_flatten = Flatten()(lstm_output)
    flatten = Dense(512, activation='relu', kernel_regularizer='l2')(lstm_output_flatten)

    action_logit = Dense(action_space, kernel_regularizer='l2')(flatten)
    value = Dense(1, kernel_regularizer='l2')(flatten)
        
    model = tf.keras.Model(inputs={'input_': input_, 'memory_state': memory_state, 'carry_state': carry_state}, 
                                outputs={'action_logit': action_logit, 'value':value, 'final_memory_state': final_memory_state, 
                                           'final_carry_state':final_carry_state})

    return model


def make_model(name, action_space):
    input_ = tf.keras.Input(shape=(128, 128, 5))
    memory_state = tf.keras.Input(shape=(128))
    carry_state = tf.keras.Input(shape=(128))

    action_logit, value, final_memory_state, final_carry_state = DeepSoccerModel(action_space)(input_, memory_state, carry_state)

    model = tf.keras.Model(inputs={'input_': input_, 'memory_state': memory_state, 'carry_state': carry_state}, 
                                outputs={'action_logit': action_logit, 'value':value, 'final_memory_state': final_memory_state, 
                                           'final_carry_state':final_carry_state}, name=name)
    return model


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
    def __init__(self, env_name, workspace_path):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        self.env = StartOpenAI_ROS_Environment(env_name)
        self.state_size = (128,128,5)
        self.action_size = self.env.action_space.n
        self.EPISODES, self.episode, self.max_average_score, self.min_average_loss = 2000000, 0, 10.0, 0.5 # specific for pong

        # Instantiate plot memory
        self.scores, self.episodes, self.average = [], [], []

        self.workspace_path = workspace_path
        
        # Create Actor-Critic network model
        self.ActorCritic = get_model(self.action_size)
        
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.writer = tf.summary.create_file_writer(workspace_path + 'src/train/tboard')

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
            action_logits = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            batch_size = states.shape[0]
            for i in range(0, batch_size):
                state = tf.expand_dims(states[i,:,:,:], 0)
                model_input = {'input_': state, 'memory_state': memory_state, 'carry_state': carry_state}
                prediction = self.ActorCritic(model_input, training=True)
                action_logit = prediction['action_logit']
                memory_state = prediction['final_memory_state']
                carry_state = prediction['final_carry_state']

                action_logits = action_logits.write(i, action_logit[0])
            
            action_logits = action_logits.stack()

            actions_onehot = tf.one_hot(actions, self.action_size)
            action_probs = tf.nn.softmax(action_logits)

            action_loss = cce_loss(actions_onehot, action_probs)
            regularization_loss = tf.reduce_sum(self.ActorCritic.losses)

            total_loss = action_loss + 1e-5 * regularization_loss
      
        grads = tape.gradient(total_loss, self.ActorCritic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.ActorCritic.trainable_variables))

        return total_loss, memory_state, carry_state
        
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
        self.ActorCritic.load_weights(self.workspace_path + 'src/train/deepsoccer_single/' + model_name)

    def save(self, model_name):
        #print("save_model")
        #self.ActorCritic.save_weights(self.workspace_path + 'src/train/deepsoccer_single/' + model_name)
        self.ActorCritic.save(self.workspace_path + 'src/train/deepsoccer_single/' + model_name) 
    
    def sl_train(self):
        #self.load('sl_model')
        episodes_loss = []
        total_step = 0
        for i in range(0, 1000):
            file_list = glob.glob(self.workspace_path + "/human_data/*.avi")
            for file in file_list:
                file_name = file.split('/')[-1].split('.')[0]

                # Read camera frame data
                try:
                    cap = cv2.VideoCapture(self.workspace_path + '/human_data/' + file_name + '.avi')
                except:
                    continue

                frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fc = 0
                ret = True

                try:
                    data = np.load(self.workspace_path + '/human_data/' + file_name + ".npy", allow_pickle=True)
                    data_0 = np.reshape(data, 1)
                    data_1 = data_0[0]
                except:
                    continue

                states, actions = [], []
                memory_state = tf.zeros([1,128], dtype=np.float32)
                carry_state = tf.zeros([1,128], dtype=np.float32)
                while fc < frameCount and ret:
                    #print("fc: ", fc)
                    ret, image_frame = cap.read()
                    image_frame_resized = cv2.resize(image_frame, (128,128), interpolation=cv2.INTER_AREA)
                    #cv2.imshow("image_frame_resized", image_frame_resized)
                    #cv2.waitKey(1)

                    frame_state_channel = image_frame_resized / 255.0
                    lidar_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * data_1['state']['lidar'][fc] / 12
                    infrared_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * data_1['state']['infrared'][fc] / 2.0

                    state_channel = np.concatenate((frame_state_channel, lidar_state_channel), axis=2)
                    state_input = np.concatenate((state_channel, infrared_state_channel), axis=2)
                    state_input = np.reshape(state_input, (1,128,128,5))

                    states.append(state_input)
                    actions.append(data_1['action'][fc])

                    fc += 1
                    total_step += 1
                    if fc % 8 == 0:
                        loss, next_memory_state, next_carry_state = self.sl_replay(states, actions, memory_state, carry_state)
                        episodes_loss.append(loss)

                        with self.writer.as_default():
                            # other model code would go here
                            tf.summary.scalar("loss", loss, step=total_step)
                            self.writer.flush()

                        memory_state = next_memory_state
                        carry_state = next_carry_state

                        average_loss = sum(episodes_loss[-50:]) / len(episodes_loss[-50:])

                        # saving best models
                        if average_loss <= self.min_average_loss:
                            self.min_average_loss = average_loss
                            self.save('sl' + '_' + str(total_step))
                            SAVING = "SAVING"
                        else:
                            SAVING = ""

                        print("episode: {}/{}, average_loss: {:.2f} {}".format(self.episode, self.EPISODES, average_loss, SAVING))
                        if(self.episode < self.EPISODES):
                            self.episode += 1
                        
                        states, actions = [], []

    def rl_train(self):
        self.load(pretrained_model)

        episodes_score = []
        total_step = 0
        while self.episode < self.EPISODES:
            # Reset episode
            score, done, SAVING = 0, False, ''
            state = self.env.reset()

            image_frame_resized = cv2.resize(state[0], (128,128), interpolation=cv2.INTER_AREA)
            image_frame_bgr = cv2.cvtColor(image_frame_resized, cv2.COLOR_RGB2BGR)
            frame_state_channel = image_frame_bgr / 255.0
            lidar_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * state[1] / 12
            infrared_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * state[2] / 2.0

            state_channel = np.concatenate((frame_state_channel, lidar_state_channel), axis=2)
            state_input = np.concatenate((state_channel, infrared_state_channel), axis=2)
            state_input = np.reshape(state_input, (1,128,128,5))

            states, actions, rewards = [], [], []
            memory_state = tf.zeros([1,128], dtype=np.float32)
            carry_state = tf.zeros([1,128], dtype=np.float32)

            initial_memory_state = memory_state
            initial_carry_state = carry_state
            for i in range(0, 800):
            	#print("i: ", i)
                action, next_memory_state, next_carry_state = agent.act(state_input, memory_state, carry_state)
                next_state, reward, done, info = self.env.step(action)
                next_image_frame_resized = cv2.resize(next_state[0], (128,128), interpolation=cv2.INTER_AREA)
                next_image_frame_bgr = cv2.cvtColor(next_image_frame_resized, cv2.COLOR_RGB2BGR)
                #cv2.imshow("image_frame_resized", image_frame_resized)
                #cv2.waitKey(1)
                frame_next_state_channel = next_image_frame_bgr / 255.0
                lidar_next_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * next_state[1] / 12
                infrared_next_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * next_state[2] / 2.0

                next_state_channel = np.concatenate((frame_next_state_channel, lidar_next_state_channel), axis=2)
                next_state_input = np.concatenate((next_state_channel, infrared_next_state_channel), axis=2)
                next_state_input = np.reshape(next_state_input, (1,128,128,5))

                states.append(state_input)
                actions.append(action)
                rewards.append(reward)

                score += reward
                state_input = next_state_input

                memory_state = next_memory_state
                carry_state = next_carry_state
                if done == True:
                    #print("done: ", done)
                    break

                total_step += 1
                if i % 8 == 0:
                    next_memory_state, next_carry_state = self.rl_replay(states, actions, rewards, 
                                                                                     initial_memory_state, initial_carry_state)
                    states, actions, rewards = [], [], []
                    
                    with self.writer.as_default():
                        # other model code would go here
                        tf.summary.scalar("score", score, step=total_step)
                        self.writer.flush()

                    initial_memory_state = memory_state
                    initial_carry_state =  carry_state

            episodes_score.append(score)

            # Update episode count
            average_score = sum(episodes_score[-50:]) / len(episodes_score[-50:])

            # saving best models
            if average_score >= self.max_average_score:
                self.max_average_score = average_score
                self.save('rl_model_' + str(total_step))
                SAVING = "SAVING"
            else:
                SAVING = ""

            print("episode: {}/{}, score: {}, average_reward: {:.2f} {}".format(self.episode, self.EPISODES, score, average_score, SAVING))
            if(self.episode < self.EPISODES):
                self.episode += 1

        env.close()            

    def test(self):
        self.load(pretrained_model)

        episodes_score = []
        total_step = 0
        while self.episode < self.EPISODES:
            self.ActorCritic.save_model(self.workspace_path + 'src/train/deepsoccer_single/')

            # Reset episode
            score, done = 0, False
            state = self.env.reset()

            image_frame_resized = cv2.resize(state[0], (128,128), interpolation=cv2.INTER_AREA)
            image_frame_bgr = cv2.cvtColor(image_frame_resized, cv2.COLOR_RGB2BGR)
            frame_state_channel = image_frame_bgr / 255.0
            lidar_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * state[1] / 12
            infrared_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * state[2] / 2.0

            state_channel = np.concatenate((frame_state_channel, lidar_state_channel), axis=2)
            state_input = np.concatenate((state_channel, infrared_state_channel), axis=2)
            state_input = np.reshape(state_input, (1,128,128,5))

            memory_state = tf.zeros([1,128], dtype=np.float32)
            carry_state = tf.zeros([1,128], dtype=np.float32)

            initial_memory_state = memory_state
            initial_carry_state = carry_state
            for i in range(0, 800):
                action, next_memory_state, next_carry_state = agent.act(state_input, memory_state, carry_state)
                next_state, reward, done, info = self.env.step(action)
                next_image_frame_resized = cv2.resize(next_state[0], (128,128), interpolation=cv2.INTER_AREA)
                next_image_frame_bgr = cv2.cvtColor(next_image_frame_resized, cv2.COLOR_RGB2BGR)
                frame_next_state_channel = next_image_frame_bgr / 255.0
                lidar_next_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * next_state[1] / 12
                infrared_next_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * next_state[2] / 2.0

                next_state_channel = np.concatenate((frame_next_state_channel, lidar_next_state_channel), axis=2)
                next_state_input = np.concatenate((next_state_channel, infrared_next_state_channel), axis=2)
                next_state_input = np.reshape(next_state_input, (1,128,128,5))

                score += reward
                state_input = next_state_input

                memory_state = next_memory_state
                carry_state = next_carry_state
                if done == True:
                    break

            episodes_score.append(score)

            # Update episode count
            average_score = sum(episodes_score[-50:]) / len(episodes_score[-50:])

            print("episode: {}/{}, score: {}, average_reward: {:.2f}".format(self.episode, self.EPISODES, score, average_score))
            if(self.episode < self.EPISODES):
                self.episode += 1

        env.close()     


env_name = task_and_robot_environment_name
agent = A2CAgent(env_name, workspace_path)

if testing == True:
    agent.test()
else:
    if sl_training == True:
        agent.sl_train()
        agent.rl_train()
    elif sl_training == False:
        agent.rl_train()