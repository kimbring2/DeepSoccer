import numpy as np
import gym


class SmartDiscrete:
    def __init__(self, ignore_keys=None, always_attack=0):
        if ignore_keys is None:
            ignore_keys = []
        self.always_attack = always_attack
        self.angle = 5
        self.all_actions_dict = {
            "[('attack', {}), ('back', 0), ('camera', [0, 0]), ('forward', 1), ('jump', 0), ('left', 0), ('right', 0), ('sneak', 0), ('sprint', 0)]".format(always_attack): 0,
            "[('attack', {}), ('back', 0), ('camera', [0, {}]), ('forward', 0), ('jump', 0), ('left', 0), ('right', 0), ('sneak', 0), ('sprint', 0)]".format(always_attack, self.angle): 1,
            "[('attack', 1), ('back', 0), ('camera', [0, 0]), ('forward', 0), ('jump', 0), ('left', 0), ('right', 0), ('sneak', 0), ('sprint', 0)]": 2,
            "[('attack', {}), ('back', 0), ('camera', [{}, 0]), ('forward', 0), ('jump', 0), ('left', 0), ('right', 0), ('sneak', 0), ('sprint', 0)]".format(always_attack, self.angle): 3,
            "[('attack', {}), ('back', 0), ('camera', [-{}, 0]), ('forward', 0), ('jump', 0), ('left', 0), ('right', 0), ('sneak', 0), ('sprint', 0)]".format(always_attack, self.angle): 4,
            "[('attack', {}), ('back', 0), ('camera', [0, -{}]), ('forward', 0), ('jump', 0), ('left', 0), ('right', 0), ('sneak', 0), ('sprint', 0)]".format(always_attack, self.angle): 5,
            "[('attack', {}), ('back', 0), ('camera', [0, 0]), ('forward', 1), ('jump', 1), ('left', 0), ('right', 0), ('sneak', 0), ('sprint', 0)]".format(always_attack): 6,
            "[('attack', {}), ('back', 0), ('camera', [0, 0]), ('forward', 0), ('jump', 0), ('left', 1), ('right', 0), ('sneak', 0), ('sprint', 0)]".format(always_attack): 7,
            "[('attack', {}), ('back', 0), ('camera', [0, 0]), ('forward', 0), ('jump', 0), ('left', 0), ('right', 1), ('sneak', 0), ('sprint', 0)]".format(always_attack): 8,
            "[('attack', {}), ('back', 1), ('camera', [0, 0]), ('forward', 0), ('jump', 0), ('left', 0), ('right', 0), ('sneak', 0), ('sprint', 0)]".format(always_attack): 9,
            "[('attack', 1), ('back', 0), ('camera', [0, 0]), ('forward', 1), ('jump', 1), ('left', 0), ('right', 0), ('sneak', 0), ('sprint', 0)]".format(always_attack): 10}
        self.ignore_keys = ignore_keys
        self.key_to_dict = {
            0: {'attack': always_attack,'back': 0,'camera': [0, 0],'forward': 1,'jump': 0,'left': 0,'right': 0,'sneak': 0,'sprint': 0},
            1: {'attack': always_attack, 'back': 0, 'camera': [0, self.angle], 'forward': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0},
            2: {'attack': 1, 'back': 0, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0},
            3: {'attack': always_attack, 'back': 0, 'camera': [self.angle, 0], 'forward': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0},
            4: {'attack': always_attack, 'back': 0, 'camera': [-self.angle, 0], 'forward': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0},
            5: {'attack': always_attack, 'back': 0, 'camera': [0, -self.angle], 'forward': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0},
            6: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 1, 'jump': 1, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0},
            7: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 1, 'right': 0, 'sneak': 0, 'sprint': 0},
            8: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 0, 'right': 1, 'sneak': 0, 'sprint': 0},
            9: {'attack': always_attack, 'back': 1, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0},
            10: {'attack': 1, 'back': 0, 'camera': [0, 0], 'forward': 1, 'jump': 1, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0},
            }

    @staticmethod
    def discrete_camera(camera):
        result = list(camera)
        if abs(result[1]) >= abs(result[0]):
            result[0] = 0
        else:
            result[1] = 0

        def cut(value, max_value=1.2):
            sign = -1 if value < 0 else 1
            if abs(value) >= max_value:
                return 5 * sign
            else:
                return 0

        cutten = list(map(cut, result))
        return cutten

    def preprocess_action_dict(self, action_dict):
        no_action_part = ["sneak", "sprint"]
        action_part = ["attack"] if self.always_attack else []
        moving_actions = ["forward", "back", "right", "left"]
        if action_dict["camera"] != [0, 0]:
            no_action_part.append("attack")
            no_action_part.append("jump")
            no_action_part += moving_actions
        elif action_dict["jump"] == 1:
            action_dict["forward"] = 1
            no_action_part += filter(lambda x: x != "forward", moving_actions)
        else:
            for a in moving_actions:
                if action_dict[a] == 1:
                    no_action_part += filter(lambda x: x != a, moving_actions)
                    no_action_part.append("attack")
                    no_action_part.append("jump")
                    break
        if "attack" not in no_action_part:
            action_dict["attack"] = 1
        for a in no_action_part:
            action_dict[a] = 0
        for a in action_part:
            action_dict[a] = 1
        return action_dict

    @staticmethod
    def dict_to_sorted_str(dict_):
        return str(sorted(dict_.items()))

    def get_key_by_action_dict(self, action_dict):
        for ignored_key in self.ignore_keys:
            action_dict.pop(ignored_key, None)
        str_dict = self.dict_to_sorted_str(action_dict)
        return self.all_actions_dict[str_dict]

    def get_action_dict_by_key(self, key):
        return self.key_to_dict[key]

    def get_actions_dim(self):
        return len(self.key_to_dict)


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
                'demo': {'dtype': 'float32'}
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