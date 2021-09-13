#!/usr/bin/env python
import rospy
import time

import RPi.GPIO as GPIO

from std_msgs.msg import String

# Pin Definitions
output_pin_1 = 5  # BCM pin 18, BOARD pin 12
output_pin_2 = 6  # BCM pin 18, BOARD pin 12


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
		GPIO.setmode(GPIO.BCM)  # BCM pin-numbering scheme from Raspberry Pi
		GPIO.setup(output_pin_1, GPIO.OUT, initial=GPIO.HIGH)
		GPIO.setup(output_pin_2, GPIO.OUT, initial=GPIO.LOW)
	elif msg.data.lower() == "stop":
		#all_stop()
		GPIO.cleanup()
	else:
		rospy.logerror(rospy.get_caller_id() + ' invalid cmd_str=%s', msg.data)


# initialization
if __name__ == '__main__':
	GPIO.setmode(GPIO.BCM)  # BCM pin-numbering scheme from Raspberry Pi
    
	GPIO.setup(output_pin_1, GPIO.OUT, initial=GPIO.HIGH)
	GPIO.setup(output_pin_2, GPIO.OUT, initial=GPIO.HIGH)
    
	# setup ros node
	rospy.init_node('deepsoccer_roller')
	
	rospy.Subscriber('~cmd_dir', String, on_cmd_dir)
	rospy.Subscriber('~cmd_raw', String, on_cmd_raw)
	rospy.Subscriber('~cmd_str', String, on_cmd_str)

	# start running
	rospy.spin()

