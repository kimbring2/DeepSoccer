from copy import deepcopy
import minerl
import numpy as np
import cv2


def reflect_image(image):
    """
    reflecting single image
    :param image: MineRL image observation
    :return: reflected image
    """
    return np.flip(image, -2)


def reflect_action(action_):
    """
    reflecting action
    swap left and right key
    reflecting pitch ['camera'][1]
    :param action_: MineRL action dict
    :return:
    """
    action_['left'], action_['right'] = action_['right'], action_['left']
    action_['camera'][1] *= -1.0

    return action_


def reflect_dict(data_dict):
    """
    reflecting data dict
    :param data_dict: MineRL data tuple [observation_dict, action_dict,
        reward, next_observation_dict, done, meta(optional)]
    :return:
    """

    # head: observation_dict(0), action_dict(1), reward_seq(2), next_observation_dict(3)
    head = deepcopy(data_dict[:4])
    tail = deepcopy(data_dict[4:])

    head[0]['pov'] = reflect_image(head[0]['pov'])
    head[1] = reflect_action(head[1])
    head[3]['pov'] = reflect_image(head[3]['pov'])

    return head + tail


def main():
    stream_name = "v1_content_squash_angel-3_16074-17640"
    env_name = "MineRLTreechop-v0"
    data = minerl.data.make(env_name)
    data_frames = list(data.load_data(stream_name, include_metadata=True))

    data_item = data_frames[39]

    data_item_reflected = reflect_dict(data_frames[39])

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import imshow

    imshow(np.concatenate([data_item[0]['pov'], data_item_reflected[0]['pov']], axis=1))
    plt.title = str(data_item[1])
    plt.show()
    print("action:", data_item[1])
    print("reflected action:", data_item_reflected[1])


if __name__ == '__main__':
    main()
