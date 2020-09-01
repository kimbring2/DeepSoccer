import os
import random
import timeit
import glob
from collections import deque

import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
from utils.tf_util import huber_loss, take_vector_elements

from collections import deque

from scipy import stats
import cv2
#from chainerrl.wrappers.atari_wrappers import LazyFrames
#from utils.discretization import SmartDiscrete

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose


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


class Agent:
    def __init__(self, config, replay_buffer, build_model, obs_space, act_space,
                 dtype_dict=None, log_freq=10):
        # global
        self.frames_to_update = config['frames_to_update']
        self.save_dir = config['save_dir']
        self.update_quantity = config['update_quantity']
        self.update_target_net_mod = config['update_target_net_mod']
        self.batch_size = config['batch_size']
        self.margin = np.array(config['margin']).astype('float32')
        self.replay_start_size = config['replay_start_size']
        self.gamma = config['gamma']
        self.learning_rate = config['learning_rate']
        self.reg = config['reg'] if 'reg' in config else 1e-5
        self.n_deque = deque([], maxlen=config['n_step'])

        self.replay_buff = replay_buffer
        self.priorities_store = list()

        if dtype_dict is not None:
            ds = tf.data.Dataset.from_generator(self.sample_generator, output_types=dtype_dict)
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
            self.sampler = ds.take
        else:
            self.sampler = self.sample_generator

        self.writer = tf.summary.create_file_writer("/home/kimbring2/catkin_ws/src/my_deepsoccer_training/src/train/tboard/")
        #print("obs_space['lidar'].shape: " + str(obs_space['lidar'].shape))
        #print("obs_space['camera']: " + str(obs_space['camera']))
        #print("obs_space['camera'].shape: " + str(obs_space['camera'].shape))
        #print("type(obs_space['camera'].shape): " + str(type(obs_space['camera'].shape)))
        #print("obs_space['infrared'].shape: " + str(obs_space['infrared'].shape))
        #print("obs_space['camera'].shape: " + str(obs_space['camera'].shape))

        self.online_model = build_model('Online_Model', obs_space, act_space, self.reg)
        self.target_model = build_model('Target_Model', obs_space, act_space, self.reg)

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self._run_time_deque = deque(maxlen=log_freq)

        self._schedule_dict = dict()
        self._schedule_dict[self.target_update] = self.update_target_net_mod
        self._schedule_dict[self.update_log] = log_freq

        self.avg_metrics = dict()
        self.action_dim = act_space.n

    def train(self, env, episodes=200, seeds=None, name="max_model.ckpt", save_mod=1,
               epsilon=0.1, final_epsilon=0.01, eps_decay=0.99, save_window=1):
        scores, counter = [], 0
        max_reward = -np.inf
        window = deque([], maxlen=save_window)
        for e in range(episodes):
            #print("self.save_dir: " + str(self.save_dir))
            #self.save(os.path.join(self.save_dir, "{}_model.ckpt".format(e)))
            #self.update_log()

            score, counter = self.train_episode(env, seeds, counter, epsilon)
            if self.replay_buff.get_stored_size() > self.replay_start_size:
                epsilon = max(final_epsilon, epsilon * eps_decay)

            scores.append(score)
            window.append(score)
            print("episode: {}  score: {}  counter: {}  epsilon: {}  max: {}"
                  .format(e, score, counter, epsilon, max_reward))

            tf.summary.scalar("reward", score, step=e)
            self.writer.flush()

            avg_reward = sum(window) / len(window)
            if avg_reward >= max_reward:
                print("MaxAvg reward moved from {:.2f} to {:.2f} (save model)".format(max_reward,
                                                                                      avg_reward))
                max_reward = avg_reward
                self.save(os.path.join(self.save_dir, name))

            if e % save_mod == 0:
                self.save(os.path.join(self.save_dir, "{}_model.ckpt".format(e)))

        return scores, counter

    def train_episode(self, env, seeds=None, current_step=0, epsilon=0.0):
        counter = current_step
        if current_step == 0:
            self.target_update()

        if seeds:
            env.seed(random.choice(seeds))

        done, score, state = False, 0, env.reset()
        reset_pose()
        while done is False:
            action = self.choose_act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            score += reward

            data_dict = {"to_demo": 1, "state": state, "action": action, "reward": reward, 
                              "next_state": next_state, "done": done, "demo": 0}
            self.perceive(data_dict)

            #self.perceive(to_demo=0, state=state, action=action, reward=reward, next_state=next_state,
            #                done=done, demo=False)
            counter += 1
            state = next_state
            if self.replay_buff.get_stored_size() > self.replay_start_size \
                    and counter % self.frames_to_update == 0:
                self.update(self.update_quantity)

        return score, counter

    def test(self, env, name="train/max_model.ckpt", number_of_trials=1, render=False):
        if name:
            self.load(name)

        total_reward = 0
        for trial_index in range(number_of_trials):
            reward = 0
            done = False
            observation = env.reset()
            rewards_dict = {}

            while not done:
                action = self.choose_act(observation)
                observation, r, done, _ = env.step(action)
                if render:
                    env.render()

                if int(r) not in rewards_dict:
                    rewards_dict[int(r)] = 0

                rewards_dict[int(r)] += 1
                reward += r

            total_reward += reward

        env.reset()

        return total_reward

    def pre_train(self, steps=150000):
        """
        pre_train phase in ForgER alg.
        :return:
        """
        print('Pre-training ...')
        self.target_update()
        self.update(steps)
        self.save(os.path.join(self.save_dir, "pre_trained_model.ckpt"))
        print('All pre-train finish.')

    def update(self, steps):
        start_time = timeit.default_timer()
        for batch in self.sampler(steps):
            indexes = batch.pop('indexes')
            #priorities = self.q_network_update(gamma=self.gamma, **batch)
            priorities = self.q_network_update(gamma=self.gamma, state=batch['state'], action=batch['action'], next_state=batch['next_state'], 
                done=batch['done'], reward=batch['reward'], demo=batch['demo'], n_state=batch['n_state'], n_done=batch['n_done'], n_reward=batch['n_reward'], 
                actual_n=batch['actual_n'], weights=batch['weights'])
            #state, action, next_state, done, reward, demo, n_state, n_done, n_reward, actual_n, weights, gamma

            self.schedule()
            self.priorities_store.append({'indexes': indexes.numpy(), 'priorities': priorities.numpy()})

            stop_time = timeit.default_timer()
            self._run_time_deque.append(stop_time - start_time)
            start_time = timeit.default_timer()

        while len(self.priorities_store) > 0:
            #print("len(self.priorities_store): " + str(len(self.priorities_store)))
            priorities = self.priorities_store.pop(0)
            self.replay_buff.update_priorities(**priorities)

    def sample_generator(self, steps=None):
        steps_done = 0
        finite_loop = bool(steps)
        steps = steps if finite_loop else 1
        while steps_done < steps:
            yield self.replay_buff.sample(self.batch_size)
            if len(self.priorities_store) > 0:
                #print("len(self.priorities_store): " + str(len(self.priorities_store)))
                priorities = self.priorities_store.pop(0)
                self.replay_buff.update_priorities(**priorities)

            steps += int(finite_loop)

    @tf.function
    def q_network_update(self, state, action, next_state, done, reward, demo,
                           n_state, n_done, n_reward, actual_n, weights,
                           gamma):
        print("Q-nn_update tracing")
        online_variables = self.online_model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(online_variables)
            q_value = self.online_model(state, training=True)
            margin = self.margin_loss(q_value, action, demo, weights)
            self.update_metrics('margin', margin)

            q_value = take_vector_elements(q_value, action)

            td_loss = self.td_loss(q_value, next_state, done, reward, 1, gamma)
            huber_td = huber_loss(td_loss, delta=0.4)
            mean_td = tf.reduce_mean(huber_td * weights)
            self.update_metrics('TD', mean_td)

            ntd_loss = self.td_loss(q_value, n_state, n_done, n_reward, actual_n, gamma)
            huber_ntd = huber_loss(ntd_loss, delta=0.4)
            mean_ntd = tf.reduce_mean(huber_ntd * weights)
            self.update_metrics('nTD', mean_ntd)

            l2 = tf.add_n(self.online_model.losses)
            self.update_metrics('l2', l2)

            all_losses = mean_td + mean_ntd + l2 + margin
            self.update_metrics('all_losses', all_losses)

        gradients = tape.gradient(all_losses, online_variables)
        # for i, g in enumerate(gradients):
        #     gradients[i] = tf.clip_by_norm(g, 10)
        self.optimizer.apply_gradients(zip(gradients, online_variables))
        priorities = tf.abs(td_loss)

        return priorities

    def td_loss(self, q_value, n_state, n_done, n_reward, actual_n, gamma):
        n_target = self.compute_target(n_state, n_done, n_reward, actual_n, gamma)
        n_target = tf.stop_gradient(n_target)
        ntd_loss = q_value - n_target
        return ntd_loss

    def compute_target(self, next_state, done, reward, actual_n, gamma):
        print("Compute_target tracing")
        q_network = self.online_model(next_state, training=True)
        argmax_actions = tf.argmax(q_network, axis=1, output_type='int32')
        q_target = self.target_model(next_state, training=True)
        target = take_vector_elements(q_target, argmax_actions)
        target = tf.where(done, tf.zeros_like(target), target)
        target = target * gamma ** actual_n
        target = target + reward

        return target

    def margin_loss(self, q_value, action, demo, weights):
        ae = tf.one_hot(action, self.action_dim, on_value=0.0,
                        off_value=self.margin)
        ae = tf.cast(ae, 'float32')
        max_value = tf.reduce_max(q_value + ae, axis=1)
        ae = tf.one_hot(action, self.action_dim)
        j_e = tf.abs(tf.reduce_sum(q_value * ae, axis=1) - max_value)
        j_e = tf.reduce_mean(j_e * weights * demo)
        
        return j_e

    def add_demo(self, expert_data=1, fixed_reward=None):
        threshold = 25
        all_data = 0
        progress = tqdm(total=self.replay_buff.get_buffer_size())
        #for l in range(0, 20):
        #    progress.update(1)

        file_list = glob.glob("/home/kimbring2/catkin_ws/src/my_deepsoccer_training/human_data/*.avi")
        #print("file_list: " + str(file_list))

        for file in glob.glob("/home/kimbring2/catkin_ws/src/my_deepsoccer_training/human_data/*.avi"):
            #print(file)
            file_name = file.split('/')[-1].split('.')[0]
            #print("file_name: " + str(file_name))

            # Read camera frame data
            cap = cv2.VideoCapture('/home/kimbring2/catkin_ws/src/my_deepsoccer_training/human_data/' + file_name + '.avi')
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            #print("frameCount: " + str(frameCount))
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #print("frameWidth: " + str(frameWidth))
            #print("frameHeight: " + str(frameHeight))
            buf = np.empty((frameCount, 128, 128, 3), np.dtype('uint8'))
            fc = 0
            ret = True
            while (fc < frameCount and ret):
                ret, image_frame = cap.read()
                image_frame_resized = cv2.resize(image_frame, (128, 128), interpolation=cv2.INTER_AREA)
                #print("image_frame_resized.shape: " + str(image_frame_resized.shape))
                buf[fc] = image_frame_resized
                fc += 1

            state = []
            next_state = []

            # Read another data
            data = np.load("/home/kimbring2/catkin_ws/src/my_deepsoccer_training/human_data/" + file_name + ".npy", allow_pickle=True)
            data_0 = np.reshape(data, 1)
            data_1 = data_0[0]
            #print("len(data_1['state']['lidar']): " + str(len(data_1['state']['lidar'])))
            for m in range(0, len(data_1['state']['lidar']) - 1):
                frame_state_channel = buf[m]
                frame_next_state_channel = buf[m+1]

                lidar_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * data_1['state']['lidar'][m] / 12
                infrared_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * data_1['state']['infrared'][m] / 2.0
                lidar_next_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * data_1['next_state']['lidar'][m] / 12
                infrared_next_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * data_1['next_state']['infrared'][m] / 2.0

                state_channel1 = np.concatenate((frame_state_channel, lidar_state_channel), axis=2)
                state_channel2 = np.concatenate((state_channel1, infrared_state_channel), axis=2)
                next_state_channel1 = np.concatenate((frame_state_channel, lidar_next_state_channel), axis=2)
                next_state_channel2 = np.concatenate((next_state_channel1, infrared_next_state_channel), axis=2)

                state.append(state_channel2)
                next_state.append(next_state_channel2)

            #print("len(state): " + str(len(state)))
            #print("len(next_state): " + str(len(next_state)))

            action = data_1['action']
            reward = data_1['reward']
            done = data_1['done']

            #print("len(next_state): " + str(len(next_state)))
            #print("len(next_state[0]): " + str(len(next_state[0])))
            for k in range(0, len(next_state)):
                #print("k: " + str(k))
                #print("state[k]: " + str(state[k]))
                #print("action[k]: " + str(action[k]))
                #print("reward[k]: " + str(reward[k]))
                #print("next_state[k]: " + str(next_state[k]))
                #print("done[k]: " + str(done[k]))

                data_dict = {"to_demo": 1, "state": state[k], "action": action[k], "reward": reward[k], 
                              "next_state": next_state[k], "done": done[k], "demo": int(expert_data)}
                self.perceive(data_dict)
                progress.update(1)

        '''
        1. to_demo, n_reward, demo,n_done, actual_n, indexes, state, done, action, weights, reward, next_state
        '''
        print('demo data added to buff')
        progress.close()
        print("***********************")
        print("all data set", all_data)
        print("***********************")

    def perceive(self, kwargs):
        self.n_deque.append(kwargs)
        if len(self.n_deque) == self.n_deque.maxlen or kwargs['done']:
            while len(self.n_deque) != 0:
                n_state = self.n_deque[-1]['next_state']
                n_done = self.n_deque[-1]['done']
                n_reward = sum([t['reward'] * self.gamma ** i for i, t in enumerate(self.n_deque)])

                self.n_deque[0]['n_state'] = n_state
                self.n_deque[0]['n_reward'] = n_reward
                self.n_deque[0]['n_done'] = n_done
                self.n_deque[0]['actual_n'] = len(self.n_deque)
                self.replay_buff.add(self.n_deque.popleft())
                if n_done:
                    print("perceive break")
                    break

    def choose_act(self, state, epsilon=0.01):
        #print("state[0].shape: " + str(state[0].shape))
        #print("state[1]: " + str(state[1]))
        #print("state[2]: " + str(state[2]))

        frame_state_channel = cv2.resize(state[0], (128, 128), interpolation=cv2.INTER_AREA)
        lidar_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * state[1] / 12
        infrared_state_channel = (np.ones(shape=(128,128,1), dtype=np.float32)) * state[2] / 2.0

        state_channel1 = np.concatenate((frame_state_channel, lidar_state_channel), axis=2)
        state_channel2 = np.concatenate((state_channel1, infrared_state_channel), axis=2)
        nn_input = np.reshape(state_channel2, (1, 128, 128, 5))

        #nn_input = np.array(state)[None]
        #nn_input = nn_input[0]
        #nn_input = nn_input[0]
        #print("type(nn_input): " + str(type(nn_input)))
        #print("nn_input: " + str(nn_input))
        #print("nn_input.shape: " + str(nn_input.shape))
        q_value = self.online_model(nn_input, training=False)
        if random.random() <= epsilon:
            return random.randint(0, self.action_dim - 1)

        return np.argmax(q_value)

    def schedule(self):
        for key, value in self._schedule_dict.items():
            if tf.equal(self.optimizer.iterations % value, 0):
                key()

    def target_update(self):
        self.target_model.set_weights(self.online_model.get_weights())

    def save(self, out_dir=None):
        self.online_model.save_weights(out_dir)

    def load(self, out_dir=None):
        self.online_model.load_weights(out_dir)

    def update_log(self):
        update_frequency = len(self._run_time_deque) / sum(self._run_time_deque)
        print("LearnerEpoch({:.2f}it/sec): ".format(update_frequency), self.optimizer.iterations.numpy())
        for key, metric in self.avg_metrics.items():
            tf.summary.scalar(key, metric.result(), step=self.optimizer.iterations)
            print('  {}:     {:.5f}'.format(key, metric.result()))
            metric.reset_states()

        self.writer.flush()

    def update_metrics(self, key, value):
        if key not in self.avg_metrics:
            self.avg_metrics[key] = tf.keras.metrics.Mean(name=key, dtype=tf.float32)

        self.avg_metrics[key].update_state(value)
