# Introduction
You can see a original Jetbot related code at https://github.com/dusty-nv/jetbot_ros/tree/master/gazebo. But, there is no URDF file of Jetbot which is needed for simulating a robot in Gazebo. Thus, I change SDF file of Jetbot to URDF. 

And all code are based on ROS URDF official tutorial http://gazebosim.org/tutorials?tut=ros_urdf where you can learn how to simulate a robot in Gazebo. I just chanage a simple 3-linkage, 2-joint arm robot of tutorial to Jetbot. 

I will upload a detailed post to https://kimbring2.github.io/2019/10/26/jetbot.html. Please rereference it if you need more information about uploaded code. 

# Issue list
~There is still a problem that the robot does not move smoothly at high speed. It seems that the physical settings of the chassis and wheel are not set correctly in Gazebo. I am checking a parameter of other robot for solving that issue.~ - solved

# Python package for Reinforcement Learning
I use a tensorflow-gpu==1.13.1 for neural network part. And opencv-python, cvlib is neeed for soccer ball detection. 

# How to Build
```
cd ~/catkin_ws/src/
git clone https://github.com/kimbring2/jetbot_gazebo.git
cd ..
catkin_make
source devel/setup.bash
```
# Dependent package install
Put a 'https://github.com/kimbring2/jetbot_soccer/tree/master/spawn_robot_tools' folder to your 'catkin_ws/src' folder.

# How to view in Rviz
```
roslaunch jetbot_description jetbot_rviz.launch
```

# How to start Jetbot model and controllers using roslaunch
```
roslaunch jetbot_gazebo main.launch
```

# Soccer model path setting
You should change a some code of sdf file at jetbot_gazebo/models/RoboCup15_MSL_Field, jetbot_gazebo/models/RoboCup15_MSL_Goal, jetbot_gazebo/models/football.

It is just example line of uri. Please change all uri path for your PC environment.
```
<uri>file:///home/[your ubuntu account]/catkin_ws/src/jetbot_soccer/jetbot_gazebo/materials/scripts/gazebo.material</uri>
```

# Troubleshooting 
<img src="image/Error_Message.png" width="600">

If you get a RLException error message, type 'source devel/setup.bash' and try again.

# How to manually send a wheel velocity commands
The range of velocity that can be given to the wheel is 0 to 100.

## Left Wheel 
For robot1
```rostopic pub -1 /robot1/joint1_velocity_controller/command std_msgs/Float64 "data: 30"```

For robot2
```rostopic pub -1 /robot2/joint1_velocity_controller/command std_msgs/Float64 "data: 30"```

## Right Wheel
For robot1
```rostopic pub -1 /robot1/joint2_velocity_controller/command std_msgs/Float64 "data: 30"```

For robot2
```rostopic pub -1 /robot2/joint2_velocity_controller/command std_msgs/Float64 "data: 30"```

# Python code for Jetbot
Move to 'jetbot/jetbot_control/src/' folder and type ```python main.py```. 
It will send a velocity command to each wheel and show a camera sensor image. Furthermore, Tensorflow code for Reinforcement Learning is implemented. Jetbot is able to only learn how to track a soccer ball at now. However, I train more advanced behavior after finishing first task.

If you run a code, it will store a Tensorflow weight file at drqn folder of your workspace. 

# Future plan
First, I plan to play a soccer game using Jetbot. However, as a result of investigating the specs of the robot that can participate in offical robot soccer competiton such as Robocup, it it determined that the Jetbot hardware can not do the soccer immediately. This is because an additional part for catching and throwing the football is required.

<img src="image/09-10-34.png" width="900">

So I am currently studying how to design a robot directly using a CAD program. Still, there seems to be no significant change in Jetbot's URDF file or Gazebo simulation method, so uploaded files can be used as they are.
