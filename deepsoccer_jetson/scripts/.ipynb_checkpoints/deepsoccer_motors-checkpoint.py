#!/usr/bin/env python
import rospy
import time

import os

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

from dynamixel_sdk_package import *                    # Uses Dynamixel SDK library
from std_msgs.msg import String

# Control table address
ADDR_MX_TORQUE_ENABLE      = 24               # Control table address is different in Dynamixel model
ADDR_MX_GOAL_POSITION      = 30
ADDR_MX_PRESENT_POSITION   = 36
ADDR_MX_MOVING_SPEED       = 32

# Protocol version
PROTOCOL_VERSION            = 1.0               # See which protocol version is used in the Dynamixel

# Default setting
DXL_ID                      = 3                # Dynamixel ID : 1
BAUDRATE                    = 57600             # Dynamixel default baudrate : 57600
DEVICENAME                  = '/dev/ttyUSB0'    # Check which port is being used on your controller
                                                # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque
DXL_MINIMUM_POSITION_VALUE  = 10           # Dynamixel will rotate between this value
DXL_MAXIMUM_POSITION_VALUE  = 4000            # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
DXL_MOVING_STATUS_THRESHOLD = 20                # Dynamixel moving status threshold

index = 0
dxl_goal_position = [DXL_MINIMUM_POSITION_VALUE, DXL_MAXIMUM_POSITION_VALUE]         # Goal position

# Initialize PortHandler instance
# Set the port path
# Get methods and members of PortHandlerLinux or PortHandlerWindows
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
# Set the protocol version
# Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()

# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    getch()
    quit()

    
def on_cmd_str_wheel1(msg):
	rospy.loginfo(rospy.get_caller_id() + ' cmd_str=%s', msg.data)
	vel = int(msg.data.lower())
	print("vel: " + str(vel))
    
	if ( (vel >= 0) | (vel <= 2047) ):
		# Enable Dynamixel Torque
		dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, 1, ADDR_MX_MOVING_SPEED, vel)
		print("dxl_comm_result: " + str(dxl_comm_result))
		if dxl_comm_result != COMM_SUCCESS:
			print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
		elif dxl_error != 0:
			print("%s" % packetHandler.getRxPacketError(dxl_error))
		else:
			print("Dynamixel has been successfully connected")
	else:
		rospy.logerror(rospy.get_caller_id() + ' invalid cmd_str=%s', msg.data)
        
        
def on_cmd_str_wheel2(msg):
	rospy.loginfo(rospy.get_caller_id() + ' cmd_str=%s', msg.data)
	vel = int(msg.data.lower())
	print("vel: " + str(vel))
    
	if ( (vel >= 0) | (vel <= 2047) ):
		# Enable Dynamixel Torque
		dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, 2, ADDR_MX_MOVING_SPEED, vel)
		print("dxl_comm_result: " + str(dxl_comm_result))
		if dxl_comm_result != COMM_SUCCESS:
			print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
		elif dxl_error != 0:
			print("%s" % packetHandler.getRxPacketError(dxl_error))
		else:
			print("Dynamixel has been successfully connected")
	else:
		rospy.logerror(rospy.get_caller_id() + ' invalid cmd_str=%s', msg.data)
        
        
def on_cmd_str_wheel3(msg):
	rospy.loginfo(rospy.get_caller_id() + ' cmd_str=%s', msg.data)
	vel = int(msg.data.lower())
	print("vel: " + str(vel))
    
	if ( (vel >= 0) | (vel <= 2047) ):
		# Enable Dynamixel Torque
		dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, 3, ADDR_MX_MOVING_SPEED, vel)
		print("dxl_comm_result: " + str(dxl_comm_result))
		if dxl_comm_result != COMM_SUCCESS:
			print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
		elif dxl_error != 0:
			print("%s" % packetHandler.getRxPacketError(dxl_error))
		else:
			print("Dynamixel has been successfully connected")
	else:
		rospy.logerror(rospy.get_caller_id() + ' invalid cmd_str=%s', msg.data)
        

def on_cmd_str_wheel4(msg):
	rospy.loginfo(rospy.get_caller_id() + ' cmd_str=%s', msg.data)
	vel = int(msg.data.lower())
	print("vel: " + str(vel))
    
	if ( (vel >= 0) | (vel <= 2047) ):
		# Enable Dynamixel Torque
		dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, 4, ADDR_MX_MOVING_SPEED, vel)
		print("dxl_comm_result: " + str(dxl_comm_result))
		if dxl_comm_result != COMM_SUCCESS:
			print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
		elif dxl_error != 0:
			print("%s" % packetHandler.getRxPacketError(dxl_error))
		else:
			print("Dynamixel has been successfully connected")
	else:
		rospy.logerror(rospy.get_caller_id() + ' invalid cmd_str=%s', msg.data)
	

# initialization
if __name__ == '__main__':
	print("main loop")
	
	# setup ros node
	rospy.init_node('deepsoccer_motors')
	rospy.Subscriber('~cmd_str_wheel1', String, on_cmd_str_wheel1)
	rospy.Subscriber('~cmd_str_wheel2', String, on_cmd_str_wheel2)
	rospy.Subscriber('~cmd_str_wheel3', String, on_cmd_str_wheel3)
	rospy.Subscriber('~cmd_str_wheel4', String, on_cmd_str_wheel4)

	# start running
	rospy.spin()

