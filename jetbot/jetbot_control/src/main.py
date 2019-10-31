#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import Float64
import math

def talker():
    pub1 = rospy.Publisher('/jetbot/joint1_velocity_controller/command', Float64, queue_size=10)
    pub2 = rospy.Publisher('/jetbot/joint2_velocity_controller/command', Float64, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        velocity_1 = 20
        velocity_2 = 15
        rospy.loginfo(velocity_1)
        rospy.loginfo(velocity_2)
        pub1.publish(velocity_1)
        pub2.publish(velocity_2)
        rate.sleep()
 
if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
