You can see a original Jetbot related code at https://github.com/dusty-nv/jetbot_ros/tree/master/gazebo. But, there is no URDF file of Jetbot which is needed for simulating a robot in Gazebo. Thus, I change SDF file of Jetbot to URDF. 

And all code are based on ROS URDF official tutorial http://gazebosim.org/tutorials?tut=ros_urdf where you can learn how to simulate a robot in Gazebo. I just chanage a simple 3-linkage, 2-joint arm robot of tutorial to Jetbot. 

I will upload a detailed post to https://kimbring2.github.io/2019/10/26/jetbot.html. Please rereference it if you need more information about uploaded code. 

# How to Build
cd ~/catkin_ws/src/
git clone https://github.com/kimbring2/jetbot_gazebo.git
cd ..
catkin_make
source devel/setup.bash

# How to view in Rviz
roslaunch jetbot_description jetbot_rviz.launch

# How to view in Gazebo
roslaunch jetbot_gazebo jetbot_world.launch

# How to start the controllers using roslaunch
roslaunch jetbot_gazebo jetbot_world.launch
roslaunch jetbot_control jetbot_control.launch

# How to manually send example commands
## Left Wheel 
rostopic pub -1 /jetbot/joint2_position_controller/command std_msgs/Float64 "data: -2.2"

## Right Wheel 
rostopic pub -1 /jetbot/joint1_position_controller/command std_msgs/Float64 "data: -2.2"

