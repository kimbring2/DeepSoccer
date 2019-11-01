#! /usr/bin/env python
import rospy
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from math import atan2

from cv_bridge import CvBridge, CvBridgeError
import cv2
import math
 
#start is x:0, y:0
x = 0.0
y = 0.0
theta = 0.0     #current angle of robot
 
bridge = CvBridge()
cv2.namedWindow("Image Window", 1)


def show_image(img):
    cv2.imshow("Image Window", img)
    cv2.waitKey(3)


def image_callback(msg):
    # log some info about the image topic
    rospy.loginfo(msg.header)

    # Try to convert the ROS Image message to a CV2 Image
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError, e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    # Show the converted image
    show_image(cv_image)


def state_callback(msg):
    # print("msg.name: " + str(msg.name))
    # msg.name: ['ground_plane', 'field', 'left_goal', 'right_goal', 'football', 'jetbot']

    football_index = (msg.name).index("football")
    left_goal_index = (msg.name).index("left_goal")
    right_goal_index = (msg.name).index("right_goal")
    #print("football_index: " + str(football_index))

    football_pose = (msg.pose)[football_index]
    print("football_pose: " + str(football_pose))

    left_goal_pose = (msg.pose)[left_goal_index]
    print("left_goal_pose: " + str(left_goal_pose))

    right_goal_pose = (msg.pose)[right_goal_index]
    print("right_goal_pose: " + str(right_goal_pose))

    global x
    global y
    global theta

    x = msg.pose[1].position.x
    y = msg.pose[1].position.y
    rot_q = msg.pose[1].orientation
    (roll, pitch, theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])
 
 
rospy.init_node('jetbot')
sub_image = rospy.Subscriber("/jetbot/camera1/image_raw", Image, image_callback)
sub_state = rospy.Subscriber('/gazebo/model_states', ModelStates, state_callback)
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

pub_vel_1 = rospy.Publisher('/jetbot/joint1_velocity_controller/command', Float64, queue_size=10)
pub_vel_2 = rospy.Publisher('/jetbot/joint2_velocity_controller/command', Float64, queue_size=10)
 
speed = Twist()
 
r = rospy.Rate(4)
 
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
    pub_vel_1.publish(velocity_1)
    pub_vel_2.publish(velocity_2)
    #pub.publish(speed)
 
r.sleep()