#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import math


bridge = CvBridge()


def show_image(img):
    cv2.imshow("Image Window", img)
    cv2.waitKey(3)


def image_callback(img_msg):
    # log some info about the image topic
    rospy.loginfo(img_msg.header)

    # Try to convert the ROS Image message to a CV2 Image
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError, e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    # Flip the image 90deg
    #cv_image = cv2.transpose(cv_image)
    #cv_image = cv2.flip(cv_image,1)

    # Show the converted image
    show_image(cv_image)

cv2.namedWindow("Image Window", 1)


def jetbot():
    pub1 = rospy.Publisher('/jetbot/joint1_velocity_controller/command', Float64, queue_size=10)
    pub2 = rospy.Publisher('/jetbot/joint2_velocity_controller/command', Float64, queue_size=10)
    sub_image = rospy.Subscriber("/jetbot/camera1/image_raw", Image, image_callback)
    rospy.init_node('jetbot', anonymous=True)
    rospy.loginfo("Hello Jetbot!")
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        velocity_1 = 10
        velocity_2 = 15
        rospy.loginfo(velocity_1)
        rospy.loginfo(velocity_2)
        pub1.publish(velocity_1)
        pub2.publish(velocity_2)
        rate.sleep()
 
if __name__ == '__main__':
    try:
        jetbot()
    except rospy.ROSInterruptException:
        pass
