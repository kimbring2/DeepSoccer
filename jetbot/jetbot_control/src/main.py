#! /usr/bin/env python
import rospy
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from math import atan2

from cv_bridge import CvBridge, CvBridgeError
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import math
import numpy as np

#start is x:0, y:0
x = 0.0
y = 0.0
theta = 0.0     #current angle of robot

bridge = CvBridge()


def image_callback_1(msg):
    # log some info about the image topic
    rospy.loginfo(msg.header)

    # Try to convert the ROS Image message to a CV2 Image
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError, e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    bbox, label, conf = cv.detect_common_objects(cv_image, confidence=0.05)
    print("bbox: " + str(bbox))
    print("label: " + str(label))
    print("conf: " + str(conf))
    for obj in label:
        if obj == "sports ball":
            index = label.index(obj)
            print("index: " + str(index))
            x1 = int(bbox[index][0])
            y1 = int(bbox[index][1])
            x2 = int(bbox[index][2])
            y2 = int(bbox[index][3])
            cv2.rectangle(cv_image, (x1,y1), (x2,y2), (0, 255, 0), 2)
            #cv2.rectangle(cv_image, (bbox[index][0],bbox[index][1]), (bbox[index][2],bbox[index][3]), (0, 255, 0), 2)
       
    #output_image = draw_bbox(cv_image, bbox, label, conf)
    #plt.imshow(output_image)
    cv2.imshow("Soccer ball Detection", cv_image)

    # Show the converted image
    #cv2.imshow("Robot1 Image Window", cv_image)
    cv2.waitKey(3)


def image_callback_2(msg):
    # log some info about the image topic
    rospy.loginfo(msg.header)

    # Try to convert the ROS Image message to a CV2 Image
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError, e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    # Show the converted image
    #cv2.imshow("Robot2 Image Window", cv_image)
    #cv2.waitKey(3)


# maybe do some 'wait for service' here
reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

def state_callback(msg):
    # print("msg.name: " + str(msg.name))
    # msg.name: ['ground_plane', 'field', 'left_goal', 'right_goal', 'football', 'jetbot']

    football_index = (msg.name).index("football")
    left_goal_index = (msg.name).index("left_goal")
    right_goal_index = (msg.name).index("right_goal")
    #print("football_index: " + str(football_index))

    football_pose = (msg.pose)[football_index]
    #print("football_pose.position.x: " + str(football_pose.position.x))

    football_pose_x = football_pose.position.x
    #print("football_pose_x: " + str(football_pose_x))

    if (football_pose_x > 5):
        reset_simulation()

    left_goal_pose = (msg.pose)[left_goal_index]
    #print("left_goal_pose: " + str(left_goal_pose))

    right_goal_pose = (msg.pose)[right_goal_index]
    #print("right_goal_pose: " + str(right_goal_pose))

    global x
    global y
    global theta

    x = msg.pose[1].position.x
    y = msg.pose[1].position.y
    rot_q = msg.pose[1].orientation
    (roll, pitch, theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])
 
 
rospy.init_node('jetbot')
sub_image_1 = rospy.Subscriber("/robot1/camera1/image_raw", Image, image_callback_1)
sub_image_2 = rospy.Subscriber("/robot2/camera1/image_raw", Image, image_callback_2)
sub_state = rospy.Subscriber('/gazebo/model_states', ModelStates, state_callback)
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

pub_vel_left_1 = rospy.Publisher('/robot1/joint1_velocity_controller/command', Float64, queue_size=5)
pub_vel_right_1 = rospy.Publisher('/robot1/joint2_velocity_controller/command', Float64, queue_size=5)

pub_vel_left_2 = rospy.Publisher('/robot2/joint1_velocity_controller/command', Float64, queue_size=5)
pub_vel_right_2 = rospy.Publisher('/robot2/joint2_velocity_controller/command', Float64, queue_size=5)
 
speed = Twist()

r = rospy.Rate(2000)

goal = Point()
goal.x = -2
goal.y = -1

while not rospy.is_shutdown():
    inc_x = goal.x - x                      #distance robot to goal in x
    inc_y = goal.y - y                      #distance robot to goal in y
    angle_to_goal = atan2 (inc_y, inc_x)    #calculate angle through distance from robot to goal in x and y
    #print abs(angle_to_goal - theta)
    if abs(angle_to_goal - theta) > 0.1:    #0.1 because it too exact for a robot if both angles should be exactly 0
        speed.linear.x = 0.0
        speed.angular.z = 0.3
    else:
        speed.linear.x = 0.3                #drive towards goal
        speed.angular.z = 0.0
    
    velocity_1 = 0
    velocity_2 = 0
    #rospy.loginfo(velocity_1)
    #rospy.loginfo(velocity_2)

    pub_vel_left_1.publish(40)
    pub_vel_right_1.publish(40)

    pub_vel_left_2.publish(40)
    pub_vel_right_2.publish(40)
    #pub.publish(speed)

r.sleep()
