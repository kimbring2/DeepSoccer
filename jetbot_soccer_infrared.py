#!/usr/bin/env python
import rospy
import time
import RPi.GPIO as GPIO

from std_msgs.msg import String

# Pin Definitions
input_pin = 27  # BOARD pin 12, BCM pin 18

def getInfraredData():
    value = GPIO.input(input_pin)
    if value == GPIO.HIGH:
        value_str = "HIGH"
    else:
        value_str = "LOW"
    
    return value_str
    print("Value read from pin {} : {}".format(input_pin, value_str))

        
# initialization
if __name__ == '__main__':
    # Pin Setup:
    # Board pin-numbering scheme
    GPIO.setmode(GPIO.BCM)  # BCM pin-numbering scheme from Raspberry Pi
    GPIO.setup(input_pin, GPIO.IN)  # set pin as an input pin

	# setup ros node
    pub = rospy.Publisher('jetbot_soccer_infrared', String, queue_size=10)
    rospy.init_node('jetbot_soccer_infrared')
    r = rospy.Rate(200) # 10hz
    while not rospy.is_shutdown():
        lidar_value = getInfraredData()
        
        if lidar_value != None:
            pub.publish(str(lidar_value))
            
        r.sleep()
