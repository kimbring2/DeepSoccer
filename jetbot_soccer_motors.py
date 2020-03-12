#!/usr/bin/env python
import rospy
import time
import RPi.GPIO as GPIO

#from Adafruit_MotorHAT import Adafruit_MotorHAT
from std_msgs.msg import String

# Pin Definitions
output_pin = 18  # BOARD pin 12, BCM pin 18

# simple string commands (left/right/forward/backward/stop)
def on_cmd_str(msg):
	rospy.loginfo(rospy.get_caller_id() + ' cmd_str=%s', msg.data)

	if msg.data.lower() == "in":
		curr_value = GPIO.HIGH
		GPIO.output(output_pin, curr_value)
	elif msg.data.lower() == "out":
		curr_value = GPIO.LOW
		GPIO.output(output_pin, curr_value)
	else:
		rospy.logerror(rospy.get_caller_id() + ' invalid cmd_str=%s', msg.data)


# initialization
if __name__ == '__main__':
	# Pin Setup:
	# Board pin-numbering scheme
	GPIO.setmode(GPIO.BCM)
    
	# set pin as an output pin with optional initial state of HIGH
	GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)

	print("Starting demo now! Press CTRL+C to exit")
	curr_value = GPIO.HIGH

	# setup ros node
	rospy.init_node('jetbot_solenoid')
	rospy.Subscriber('~cmd_str', String, on_cmd_str)

	# start running
	rospy.spin()

