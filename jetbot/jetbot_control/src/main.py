#! /usr/bin/env python
############## ROS Import ###############
import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge, CvBridgeError
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import math
from math import atan2
import numpy as np
import random
import time
import itertools
import os


############## Deep Learning Import ###############
import tensorflow as tf
import tensorflow.contrib.slim as slim


############## Deep Learning Part ###############
class Qnetwork():
    def __init__(self, h_size, rnn_cell, myScope):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.scalarInput =  tf.placeholder(shape=[None,21168], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1,84,84,3])
        self.conv1 = slim.convolution2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8,8],
                                        stride=[4,4], padding='VALID', biases_initializer=None, scope=myScope + '_conv1')
        self.conv2 = slim.convolution2d(inputs=self.conv1, num_outputs=64, kernel_size=[4,4], stride=[2,2], padding='VALID',
                                        biases_initializer=None, scope=myScope + '_conv2')
        self.conv3 = slim.convolution2d(inputs=self.conv2, num_outputs=64, kernel_size=[3,3], stride=[1,1], padding='VALID',
                                        biases_initializer=None, scope=myScope + '_conv3')
        self.conv4 = slim.convolution2d(inputs=self.conv3, num_outputs=h_size, kernel_size=[7,7], stride=[1,1], padding='VALID',
                                        biases_initializer=None, scope=myScope + '_conv4')
        
        self.trainLength = tf.placeholder(dtype=tf.int32)

        # We take the output from the final convolutional layer and send it to a recurrent layer.
        # The input must be reshaped into [batch x trace x units] for rnn processing, 
        # and then returned to [batch x units] when sent through the upper levles.
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.convFlat = tf.reshape(slim.flatten(self.conv4), [self.batch_size,self.trainLength,h_size])
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.convFlat, cell=rnn_cell, dtype=tf.float32, 
                                                     initial_state=self.state_in, scope=myScope + '_rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1,h_size])

        # The output from the recurrent player is then split into separate Value and Advantage streams
        self.streamA,self.streamV = tf.split(self.rnn, 2, 1)
        self.AW = tf.Variable(tf.random_normal([h_size//2,6]))
        self.VW = tf.Variable(tf.random_normal([h_size//2,1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)
        
        self.salience = tf.gradients(self.Advantage, self.imageIn)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage,axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)
        
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 6, dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        self.td_error = tf.square(self.targetQ - self.Q)
        
        # In order to only propogate accurate gradients through the network, we will mask the first
        # half of the losses for each trace as per Lample & Chatlot 2016
        self.maskA = tf.zeros([self.batch_size, self.trainLength//2])
        self.maskB = tf.ones([self.batch_size, self.trainLength//2])
        self.mask = tf.concat([self.maskA,self.maskB], 1)
        self.mask = tf.reshape(self.mask, [-1])
        self.loss = tf.reduce_mean(self.td_error * self.mask)
        
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


class experience_buffer():
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []

        self.buffer.append(experience)
            
    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            #print("len(episode): " + str(len(episode)))
            #if (len(episode) != 51):
            #    return -1

            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append(episode[point:point + trace_length])

        sampledTraces = np.array(sampledTraces)

        return np.reshape(sampledTraces, [batch_size*trace_length,5])


# These functions allows us to update the parameters of our target network with those of the primary network.
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))

    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

    total_vars = len(tf.trainable_variables())
    a = tf.trainable_variables()[0].eval(session=sess)
    b = tf.trainable_variables()[total_vars//2].eval(session=sess)
    if a.all() == b.all():
        print("Target Set Success")
    else:
        print("Target Set Failed")


#This is a simple function to reshape our game frames.
def processState(state1):
    return np.reshape(state1, [21168])


############## Deep Learning Part ###############
timestep = 0
episodeBuffer = []
d = False
rAll = 0
j = 0

image_state = np.empty([84, 84, 3])

s = processState(image_state)
s1 = s
r = 0

############## ROS Part ###############
bridge = CvBridge()

image_j = j
def image_callback_1(msg):
    global image_state
    global image_j 
    global r

    # log some info about the image topic
    #rospy.loginfo(msg.header)

    # Try to convert the ROS Image message to a CV2 Image
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError, e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    bbox, label, conf = cv.detect_common_objects(cv_image, confidence=0.05)
    #print("bbox: " + str(bbox))
    #print("label: " + str(label))
    #print("conf: " + str(conf))
    r = 0
    for obj in label:
        if obj == "sports ball":
            index = label.index(obj)
            ball_bbox_x1 = bbox[index][0]
            ball_bbox_x2 = bbox[index][2]
            ball_bbox_middle = (ball_bbox_x1 + ball_bbox_x2) / 2.0
            #print("ball_bbox_middle: " + str(ball_bbox_middle))
            
            if ( (ball_bbox_middle > 330) | (ball_bbox_middle < 390) ):
                r = 1
            
            x1 = int(bbox[index][0])
            y1 = int(bbox[index][1])
            x2 = int(bbox[index][2])
            y2 = int(bbox[index][3])
            cv2.rectangle(cv_image, (x1,y1), (x2,y2), (0,255,0), 2)
    
    #print("cv_image.shape: " + str(cv_image.shape))
    ratio = 84.0 / cv_image.shape[1]
    dim = (84, int(cv_image.shape[0] * ratio))
 
    # perform the actual resizing of the image and show it
    new_cv_image = cv2.resize(cv_image, dim, interpolation=cv2.INTER_AREA)
    #print("new_cv_image.shape: " + str(new_cv_image.shape))

    if (image_j != j):
        #print("j of image_callback_1: " + str(j))
        image_state = new_cv_image
        
    image_j = j
    # Show the converted image
    #cv2.imshow("Robot1 Camera", cv_image)
    #cv2.waitKey(3)


def image_callback_2(msg):
    # log some info about the image topic
    #rospy.loginfo(msg.header)

    # Try to convert the ROS Image message to a CV2 Image
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError, e:
        rospy.logerr("CvBridge Error: {0}".format(e))


# maybe do some 'wait for service' here
reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

state_j = j
def state_callback(msg):
    global state_j 
    global d

    #print("msg.name: " + str(msg.name))
    # msg.name: ['ground_plane', 'field', 'left_goal', 'right_goal', 'football', 'jetbot']

    robot1_index = (msg.name).index("robot1")
    football_index = (msg.name).index("football")
    left_goal_index = (msg.name).index("left_goal")
    right_goal_index = (msg.name).index("right_goal")
    #print("football_index: " + str(football_index))

    robot1_pose = (msg.pose)[robot1_index]
    print("robot1_pose.: " + str(robot1_pose))

    football_pose = (msg.pose)[football_index]
    #print("football_pose.position.x: " + str(football_pose.position.x))

    football_pose_x = football_pose.position.x
    #print("football_pose_x: " + str(football_pose_x))

    if (football_pose_x > 10):
        d = True

    left_goal_pose = (msg.pose)[left_goal_index]
    #print("left_goal_pose: " + str(left_goal_pose))

    right_goal_pose = (msg.pose)[right_goal_index]
    #print("right_goal_pose: " + str(right_goal_pose))

    global x
    global y
    global theta

    #if (state_j != j):
    #    #print("j of state_callback: " + str(j))

    state_j = j

    x = msg.pose[1].position.x
    y = msg.pose[1].position.y
    rot_q = msg.pose[1].orientation
    (roll, pitch, theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])
 

############## ROS Part ###############
rospy.init_node('jetbot')

sub_image_1 = rospy.Subscriber("/robot1/camera1/image_raw", Image, image_callback_1)
sub_image_2 = rospy.Subscriber("/robot2/camera1/image_raw", Image, image_callback_2)
sub_state = rospy.Subscriber('/gazebo/model_states', ModelStates, state_callback)

pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

pub_vel_left_1 = rospy.Publisher('/robot1/joint1_velocity_controller/command', Float64, queue_size=5)
pub_vel_right_1 = rospy.Publisher('/robot1/joint2_velocity_controller/command', Float64, queue_size=5)

pub_vel_left_2 = rospy.Publisher('/robot2/joint1_velocity_controller/command', Float64, queue_size=5)
pub_vel_right_2 = rospy.Publisher('/robot2/joint2_velocity_controller/command', Float64, queue_size=5)
 
rate = rospy.Rate(2000)

stop_action = [0, 0]
forward_action = [40, 40]
left_action = [40, -40]
right_action = [-40, 40]
bacward_action = [-40, -40]
kick_action = [100, 100]
robot_action_list = [stop_action, forward_action, left_action, right_action, bacward_action, kick_action]

set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
pose = Pose() 
pose.position.x = np.random.randint(1,20) / 10.0
pose.position.y = np.random.randint(1,20) / 10.0
#pose.position.x = 0
#pose.position.y = 0
pose.position.z = 0.12
  
pose.orientation.x = 0
pose.orientation.y = 0
pose.orientation.z = 0
pose.orientation.w = 0
    
state_model = ModelState()   
state_model.model_name = "football"
state_model.pose = pose
resp = set_state(state_model)

############## Deep Learning Part ###############
# Setting the training parameters
batch_size = 4 # How many experience traces to use for each training step.
trace_length = 4 # How long each experience trace will be when training
update_freq = 5 # How often to perform a training step.
y = .99 # Discount factor on the target Q-values
startE = 1 # Starting chance of random action
endE = 0.1 # Final chance of random action
anneling_steps = 20000 # How many steps of training to reduce startE to endE.
#num_episodes = 100 # How many episodes of game environment to train network with.
pre_train_steps = 2000 # How many steps of random actions before training begins.
load_model = False # Whether to load a saved model.
path = "./drqn" # The path to save our model to.
h_size = 512 # The size of the final convolutional layer before splitting it into Advantage and Value streams.
#max_epLength = 50 # The max allowed length of our episode.
time_per_step = 1 # Length of each step used in gif creation
summaryLength = 100 # Number of epidoes to periodically save for analysis
tau = 0.001

state = (np.zeros([1,h_size]), np.zeros([1,h_size])) # Reset the recurrent layer's hidden state

tf.reset_default_graph()

# We define the cells for the primary and target q-networks
cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
cellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
mainQN = Qnetwork(h_size, cell, 'main')
targetQN = Qnetwork(h_size, cellT, 'target')

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=5)
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)
myBuffer = experience_buffer()

# Set the rate of random action decrease. 
e = startE
stepDrop = (startE - endE) / anneling_steps

# Create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

#Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

sess = tf.Session()

if load_model == True:
    print ('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(path)
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(init)

updateTarget(targetOps,sess)

############## ROS + Deep Learning Part ###############
while not rospy.is_shutdown():
    '''
    if np.random.rand(1) < e or total_steps < pre_train_steps:
        state1 = sess.run(mainQN.rnn_state, 
                          feed_dict = {mainQN.scalarInput:[s/255.0], mainQN.trainLength:1, mainQN.state_in:state, mainQN.batch_size:1})
        a = np.random.randint(0,6)
    else:
        a, state1 = sess.run([mainQN.predict, mainQN.rnn_state],
                              feed_dict = {mainQN.scalarInput:[s/255.0], mainQN.trainLength:1, mainQN.state_in:state, mainQN.batch_size:1})
        a = a[0]
    '''
    robot1_action = robot_action_list[2]
    #print("robot1_action: " + str(robot1_action))

    pub_vel_left_1.publish(robot1_action[0])
    pub_vel_right_1.publish(robot1_action[1])
    #pub_vel_left_1.publish(0)
    #pub_vel_right_1.publish(0)

    #robot2_action = robot_action_list[a]
    #pub_vel_left_2.publish(robot2_action[0])
    #pub_vel_right_2.publish(robot2_action[1])
    '''
    time.sleep(0.5)
    j = j + 1
    print("j: " + str(j))
    if j > 100:
        d = True
    
    s1 = processState(image_state)
    
    total_steps += 1
    episodeBuffer.append(np.reshape(np.array([s,a,r,s1,d]), [1,5]))
    print("total_steps: " + str(total_steps))
    if total_steps > pre_train_steps:
        if e > endE:
            e -= stepDrop

        if total_steps % (update_freq) == 0:
            updateTarget(targetOps, sess)

            # Reset the recurrent layer's hidden state
            state_train = (np.zeros([batch_size,h_size]), np.zeros([batch_size,h_size])) 
                    
            trainBatch = myBuffer.sample(batch_size, trace_length) # Get a random batch of experiences.
            if (trainBatch == -1):
                continue

            # Below we perform the Double-DQN update to the target Q-values
            Q1 = sess.run(mainQN.predict, feed_dict = {mainQN.scalarInput:np.vstack(trainBatch[:,3] / 255.0),
						       						   mainQN.trainLength:trace_length, mainQN.state_in:state_train, mainQN.batch_size:batch_size})

            Q2 = sess.run(targetQN.Qout, feed_dict = {targetQN.scalarInput:np.vstack(trainBatch[:,3] / 255.0),
                          							  targetQN.trainLength:trace_length, targetQN.state_in:state_train, targetQN.batch_size:batch_size})

            end_multiplier = -(trainBatch[:,4] - 1)
            doubleQ = Q2[range(batch_size * trace_length), Q1]
            targetQ = trainBatch[:,2] + (y * doubleQ * end_multiplier)

            # Update the network with our target values.
            sess.run(mainQN.updateModel, feed_dict = {mainQN.scalarInput:np.vstack(trainBatch[:,0] / 255.0), 
						       						  mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1], mainQN.trainLength:trace_length,
 						       						  mainQN.state_in:state_train, mainQN.batch_size:batch_size})

    rAll += r
    print("rAll: " + str(rAll))
    print("e: " + str(e))
    print("")

    s = s1
    state = state1
    if d == True:
        bufferArray = np.array(episodeBuffer)
        episodeBuffer = list(zip(bufferArray))
        myBuffer.add(episodeBuffer)
        jList.append(j)
        rList.append(rAll)

        timestep = 0
        episodeBuffer = []
        d = False
        rAll = 0
        j = 0

        reset_simulation()

        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        pose = Pose() 
        pose.position.x = np.random.randint(1,20) / 10.0
        pose.position.y = np.random.randint(1,20) / 10.0
        #pose.position.x = 0
        #pose.position.y = 0
        pose.position.z = 0.12
  
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 0
    
        state_model = ModelState()   
        state_model.model_name = "football"
        state_model.pose = pose
        resp = set_state(state_model)

    # Periodically save the model. 
    if total_steps % 100 == 0 and total_steps != 0:
        saver.save(sess, path + '/model-' + str(int(total_steps / 100)) + '.cptk')
        print ("Saved Model")
    '''
rate.sleep()
