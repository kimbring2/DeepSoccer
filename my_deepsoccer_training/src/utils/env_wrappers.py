import os
import random
from collections import deque

import cv2
import gym
import numpy as np
from chainerrl.wrappers.atari_wrappers import LazyFrames
mapping = dict()


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


def get_discretizer(name):
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Registered wrappers:', ', '.join(mapping.keys()))


class FrameSkip(gym.Wrapper):
    """Return every `skip`-th frame and repeat given action during skip.
    Note that this wrapper does not "maximize" over the skipped frames.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)

        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, channel_order='hwc', use_tuple=False):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.observations = deque([], maxlen=k)
        self.stack_axis = {'hwc': 2, 'chw': 0}[channel_order]
        self.use_tuple = use_tuple

        if self.use_tuple:
            pov_space = env.observation_space[0]
            inv_space = env.observation_space[1]
        else:
            pov_space = env.observation_space

        low_pov = np.repeat(pov_space.low, k, axis=self.stack_axis)
        high_pov = np.repeat(pov_space.high, k, axis=self.stack_axis)
        pov_space = gym.spaces.Box(low=low_pov, high=high_pov, dtype=pov_space.dtype)

        if self.use_tuple:
            low_inv = np.repeat(inv_space.low, k, axis=0)
            high_inv = np.repeat(inv_space.high, k, axis=0)
            inv_space = gym.spaces.Box(low=low_inv, high=high_inv, dtype=inv_space.dtype)
            self.observation_space = gym.spaces.Tuple(
                (pov_space, inv_space))
        else:
            self.observation_space = pov_space

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.observations.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.observations.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.observations) == self.k
        if self.use_tuple:
            frames = [x[0] for x in self.observations]
            inventory = [x[1] for x in self.observations]
            return (LazyFrames(list(frames), stack_axis=self.stack_axis),
                    LazyFrames(list(inventory), stack_axis=0))
        else:
            return LazyFrames(list(self.observations), stack_axis=self.stack_axis)


class ObtainPoVWrapper(gym.ObservationWrapper):
    """Obtain 'pov' value (current game display) of the original observation."""
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = self.env.observation_space.spaces['pov']

    def observation(self, observation):
        return observation['pov']


class DiscreteBase(gym.Wrapper):
    def __init__(self, env):
        super(DiscreteBase, self).__init__(env)
        self.action_dict = {}
        self.action_space = gym.spaces.Discrete(len(self.action_dict))

    def step(self, action):
        s, r, done, info = self.env.step(self.action_dict[action])
        return s, r, done, info

    def sample_action(self):
        return self.action_space.sample()



@register("Treechop")
class TreechopDiscretWrapper(DiscreteBase):
    def __init__(self, env, always_attack=1, angle=5):
        DiscreteBase.__init__(self, env)
        self.action_dict = {
            0: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 1, 'jump': 0, 'left': 0, 'right': 0,
                'sneak': 0, 'sprint': 0},
            1: {'attack': always_attack, 'back': 0, 'camera': [0, angle], 'forward': 0, 'jump': 0, 'left': 0,
                'right': 0, 'sneak': 0, 'sprint': 0},
            2: {'attack': 1, 'back': 0, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0,
                'sprint': 0},
            3: {'attack': always_attack, 'back': 0, 'camera': [angle, 0], 'forward': 0, 'jump': 0, 'left': 0,
                'right': 0, 'sneak': 0, 'sprint': 0},
            4: {'attack': always_attack, 'back': 0, 'camera': [-angle, 0], 'forward': 0, 'jump': 0, 'left': 0,
                'right': 0, 'sneak': 0, 'sprint': 0},
            5: {'attack': always_attack, 'back': 0, 'camera': [0, -angle], 'forward': 0, 'jump': 0, 'left': 0,
                'right': 0, 'sneak': 0, 'sprint': 0},
            6: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 1, 'jump': 1, 'left': 0, 'right': 0,
                'sneak': 0, 'sprint': 0},
            7: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 1, 'right': 0,
                'sneak': 0, 'sprint': 0},
            8: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 0, 'right': 1,
                'sneak': 0, 'sprint': 0},
            9: {'attack': always_attack, 'back': 1, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 0, 'right': 0,
                'sneak': 0, 'sprint': 0}}
        self.action_space = gym.spaces.Discrete(len(self.action_dict))


@register("CarRacing")
class CarRacingWrapper(DiscreteBase):
    def __init__(self, env):
        DiscreteBase.__init__(self, env)
        self.action_dict = [[0.0, 1.0, 0.0], [1.0, 0.3, 0], [-1.0, 0.3, 0.0], [0.0, 0.0, 0.8]]
        self.action_space = gym.spaces.Discrete(len(self.action_dict))


class SaveVideoWrapper(gym.Wrapper):
    current_episode = 0

    def __init__(self, env, path='train/', resize=4, reward_threshold=0):
        """
        :param env: wrapped environment
        :param path: path to save videos
        :param resize: resize factor
        """
        super().__init__(env)
        self.path = path
        self.recording = []
        self.rewards = [0]
        self.resize = resize
        self.reward_threshold = reward_threshold
        self.previous_reward = 0

    def step(self, action):
        """
        make a step in environment
        :param action: agent's action
        :return: observation, reward, done, info
        """
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        self.recording.append(self.bgr_to_rgb(observation['pov']))
        return observation, reward, done, info

    def get_reward(self):
        return sum(map(int, self.rewards))

    def reset(self, **kwargs):
        """
        reset environment and save game video if its not empty
        :param kwargs:
        :return: current observation
        """

        reward = self.get_reward()
        if self.current_episode > 0 and reward >= self.reward_threshold:
            name = str(self.current_episode).zfill(4) + "r" + str(reward).zfill(4) + ".mp4"
            full_path = os.path.join(self.path, name)
            upscaled_video = [self.upscale_image(image, self.resize) for image in self.recording]
            self.save_video(full_path, video=upscaled_video)
        self.current_episode += 1
        self.rewards = [0]
        self.recording = []
        self.env.seed(self.current_episode)
        observation = self.env.reset(**kwargs)
        self.recording.append(self.bgr_to_rgb(observation['pov']))
        return observation

    @staticmethod
    def upscale_image(image, resize):
        """
        increase image size (for better video quality)
        :param image: original image
        :param resize:
        :return:
        """
        size_x, size_y, size_z = image.shape
        return cv2.resize(image, dsize=(size_x * resize, size_y * resize))

    @staticmethod
    def save_video(filename, video):
        """
        saves video from list of np.array images
        :param filename: filename or path to file
        :param video: [image, ..., image]
        :return:
        """
        size_x, size_y, size_z = video[0].shape
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 60.0, (size_y, size_x))
        for image in video:
            out.write(image)
        out.release()
        cv2.destroyAllWindows()

    @staticmethod
    def bgr_to_rgb(image):
        """
        converts BGR image to RGB
        :param image: bgr image
        :return: rgb image
        """
        return image[..., ::-1]


class SaveSaliencyVideoWrapper(gym.Wrapper):
    current_episode = 0

    def __init__(self, env, path='train/', resize=4, saliency_map=None, frame_skip=4, frame_stack=2, name=None):
        super().__init__(env)
        self.path = path
        self.recording = []
        self.rewards = [0]
        self.resize = resize
        self.saliency_map = saliency_map
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.name = name

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        self.recording.append(observation['pov'])
        return observation, reward, done, info

    def reset(self, **kwargs):
        if self.recording:
            name = self.name + str(self.current_episode).zfill(4) + \
                   "r" + str(sum(map(int, self.rewards))).zfill(4) + ".mp4"
            print('saving to', os.path.realpath(os.path.join(self.path, name)))
            outs = [cv2.VideoWriter(os.path.join(self.path, stream + name), cv2.VideoWriter_fourcc(*'mp4v'),
                                    60.0, (64 * self.resize, 64 * self.resize)) for stream in ['a_', 'v_']]
            saliency_stacks = [[] for _ in ['a_', 'v_']]
            idx_sample = random.sample(range(self.frame_stack * self.frame_skip, len(self.recording)), 9)
            for i in range(self.frame_stack * self.frame_skip, len(self.recording)):
                obs = self.recording[i - self.frame_skip * (self.frame_stack - 1):i + 1:self.frame_skip]
                obs = np.concatenate(obs, axis=-1)
                rsls = self.saliency_map(obs)
                for rsl, out, stack in zip(rsls, outs, saliency_stacks):
                    if i in idx_sample:
                        stack.append(rsl.astype(np.uint8))
                    image = cv2.resize(rsl.astype(np.uint8), dsize=(64 * self.resize, 64 * self.resize))
                    out.write(image)
            for stacks, name in zip(saliency_stacks, ['a_', 'v_']):
                stacks = [np.hstack(tuple(stacks[i:i+3])) for i in range(0,9,3)]
                stacks = np.vstack(tuple(stacks))
                cv2.imwrite(os.path.join(self.path, name+'stack.png'), stacks)
            for out in outs: out.release()
            cv2.destroyAllWindows()

        self.current_episode += 1
        self.rewards = [0]
        self.recording = []
        observation = self.env.reset(**kwargs)
        self.recording.append(observation['pov'])

        return observation


class SaliencyVideo:
    def __init__(self, saliency_map, model_name='model.ckpt', path='train/', resize=4,
                 frame_skip=4, frame_stack=2, rows=3, columns=3):
        self.model_name = model_name
        self.path = path
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.resize = resize
        self.saliency_map = saliency_map
        self.rows = rows
        self.columns = columns

    def make_saliency_video(self, recording):
        video_name = self.model_name + ".mp4"
        outs = [cv2.VideoWriter(os.path.join(self.path, stream + video_name), cv2.VideoWriter_fourcc(*'mp4v'),
                                60.0, (64 * self.resize, 64 * self.resize)) for stream in ['a_', 'v_']]
        saliency_stacks = [[] for _ in ['a_', 'v_']]
        # idx_sample = random.sample(range(self.frame_stack * self.frame_skip, len(recording)), self.rows * self.columns)
        idx_sample = [50 * i for i in range(1, self.rows * self.columns + 1)]
        for i in range(self.frame_stack * self.frame_skip, len(recording)):
            obs = recording[i - self.frame_skip * (self.frame_stack - 1):i + 1:self.frame_skip]
            obs = np.concatenate(obs, axis=-1)
            rsls = self.saliency_map(obs)
            for rsl, out, stack in zip(rsls, outs, saliency_stacks):
                if i in idx_sample:
                    stack.append(rsl.astype(np.uint8))
                image = cv2.resize(rsl.astype(np.uint8), dsize=(64 * self.resize, 64 * self.resize))
                out.write(image)
        for stacks, name in zip(saliency_stacks, ['a_', 'v_']):
            stacks = [np.hstack(tuple(stacks[i:i+self.columns]))
                      for i in range(0, self.rows * self.columns, self.columns)]
            stacks = np.vstack(tuple(stacks))
            img_name = name + self.model_name + '_stack.png'
            cv2.imwrite(os.path.join(self.path, img_name), stacks)
        for out in outs:
            out.release()
        cv2.destroyAllWindows()


class SaveDS(gym.Wrapper):
    def __init__(self, env):
        super(SaveDS, self).__init__(env)
        self.actions, self.episode_act = list(), list()
        self.observations, self.episode_obs = list(), list()
        self.returns, self.episode_ret = list(), list()
        self.dones, self.episode_done = list(), list()
        self.scores = list()
        self.score = 0

    def step(self, action):
        obs, rew, done, _ = self.env.step(action)
        self.score += rew
        self.episode_obs.append(obs)
        self.episode_act.append(action)
        self.episode_ret.append(rew)
        self.episode_done.append(done)
        if done:
            self.scores.append(self.score)
            self.actions.append(np.array(self.episode_act))
            self.observations.append(np.array(self.episode_obs))
            self.returns.append(np.array(self.episode_ret))
            self.dones.append(np.array(self.episode_done))
            self.episode_act = list()
            self.episode_obs = list()
            self.episode_ret = list()
            self.episode_done = list()
            self.score = 0
        return obs, rew, done, _

    def reset(self):
        obs = self.env.reset()
        self.episode_obs.append(obs)
        return obs

    def save_ds(self, path):
        avg_rew = sum(self.scores) / len(self.scores)
        file_name = path + '_{:.0f}'.format(avg_rew)
        np.savez(file_name, obs=self.observations,
                 acs=self.actions, rew=self.returns,
                 done=self.dones)
