import wandb
import tensorflow as tf

wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True, anonymous='allow', project="DeepMine")
