from collections import deque

from scipy import stats
import numpy as np
from chainerrl.wrappers.atari_wrappers import LazyFrames
from utils.discretization import SmartDiscrete


class TreechopLoader:
    def __init__(self, data, frame_skip=4, frame_stack=2, always_attack=1, threshold=60):
        self.data = data
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.discrete_maker = SmartDiscrete(ignore_keys=["place", "nearbySmelt", "nearbyCraft",
                                                         "equip", "craft"],
                                            always_attack=always_attack)
        self.threshold = threshold

    def sarsd_iter(self, *args, **kwargs):
        print("args: " + str(args))
        print("kwargs: " + str(kwargs))
        #for s, a, r, _, d in self.data.batch_iter(*args, **kwargs):
        iteration = 0
        for s, a, r, _, d in self.data.batch_iter(batch_size=1, num_epochs=1, seq_len=200):
            iteration += 1
            print("iteration: " + str(iteration))

            if iteration >= 2000:
                break

            if sum(r[0]) < self.threshold:
                continue

            reward = np.sum(self._skip_stack(r), axis=1)
            done = np.any(self._skip_stack(d), axis=1)
            action = {key: self._skip_stack(value)
                      for key, value in a.items()}

            for key, value in action.items():
                if key != 'camera':
                    most_freq, _ = stats.mode(value, axis=1)
                    action[key] = np.squeeze(most_freq)
                else:
                    mean = np.mean(value, axis=-1)
                    mask = np.abs(mean) > 1.2
                    sign = np.sign(mean)
                    argmax = np.argmax(np.abs(mean), axis=1)
                    one_hot = np.eye(2)[argmax]
                    tolist = (one_hot * sign * mask * 5 + 0).astype('int').tolist()
                    action[key] = tolist

            observation = s['pov'] / 255.0
            for i in range(self.frame_skip):
                deque_state = deque([observation[i]] * self.frame_stack, maxlen=self.frame_stack)
                state = LazyFrames(list(deque_state))
                for j in range(i, len(observation) - self.frame_skip, self.frame_skip):
                    discrete_action = {key: value[j] for key, value in action.items()}
                    discrete_action = self.discrete_maker.preprocess_action_dict(discrete_action)
                    discrete_action = self.discrete_maker.get_key_by_action_dict(discrete_action)
                    deque_state.append(observation[j + self.frame_skip])
                    next_state = LazyFrames(list(deque_state))
                    yield state, discrete_action, reward[j], next_state, done[j]
                    state = next_state

        return state, action, reward, next_state, done

    def _skip_stack(self, array):
        if isinstance(array, list):
            array = np.array(array)

        length = array.shape[0]
        stack = [array[j:length-self.frame_skip+j+1] for j in range(self.frame_skip)]
        return np.stack(stack, axis=-1)


class NPZLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def sarsd_iter(self, *args, **kwargs):
        data = np.load(self.data_path, allow_pickle=True)
        data = {key: np.squeeze(np.concatenate(value, axis=0)) for key, value in data.items()}
        for obs, acs, rew, next_obs, done in zip(data['obs'][:-1],
                                                 data['acs'],
                                                 data['rew'],
                                                 data['obs'][1:],
                                                 data['done']):
            yield obs, acs, rew, next_obs, done





