#!/usr/bin/env python
import rospy
import time

import RPi.GPIO as GPIO
from std_msgs.msg import String


# sets motor speed between [-1.0, 1.0]
def set_speed(motor_ID, value):
    max_pwm = 115.0
    speed = int(min(max(abs(value * max_pwm), 0), max_pwm))
    if value > 0:
        GPIO.output(5, GPIO.HIGH)
        GPIO.output(6, GPIO.LOW)
    else:
        GPIO.output(5, GPIO.LOW)
        GPIO.output(6, GPIO.HIGH)


# stops all motors
def all_stop():
    GPIO.output(5, GPIO.HIGH)
    GPIO.output(6, GPIO.HIGH)
    

# directional commands (degree, speed)
def on_cmd_dir(msg):
    rospy.loginfo(rospy.get_caller_id() + ' cmd_dir=%s', msg.data)

    
# raw L/R motor commands (speed, speed)
def on_cmd_raw(msg):
    rospy.loginfo(rospy.get_caller_id() + ' cmd_raw=%s', msg.data)

    
# simple string commands (left/right/forward/backward/stop)
def on_cmd_str(msg):
    rospy.loginfo(rospy.get_caller_id() + ' cmd_str=%s', msg.data)

    if msg.data.lower() == "in":
        #set_speed(motor_ID,   1.0)
        GPIO.output(5, GPIO.HIGH)
        GPIO.output(6, GPIO.LOW)
    elif msg.data.lower() == "out":
        GPIO.output(5, GPIO.LOW)
        GPIO.output(6, GPIO.HIGH)
        #set_speed(motor_ID,  -1.0) 
    elif msg.data.lower() == "stop":
        all_stop()
    else:
        rospy.logerror(rospy.get_caller_id() + ' invalid cmd_str=%s', msg.data)


# initialization
if __name__ == '__main__':
    GPIO.setmode(GPIO.BCM)  # BCM pin-numbering scheme from Raspberry Pi
    
    # setup motor gpio
    GPIO.setup(5, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(6, GPIO.OUT, initial=GPIO.HIGH)

    # stop the motors as precaution
    all_stop()

    # setup ros node
    rospy.init_node('deepsoccer_roller')
    
    rospy.Subscriber('~cmd_dir', String, on_cmd_dir)
    rospy.Subscriber('~cmd_raw', String, on_cmd_raw)
    rospy.Subscriber('~cmd_str', String, on_cmd_str)

    # start running
    rospy.spin()

    # stop motors before exiting
    all_stop()

