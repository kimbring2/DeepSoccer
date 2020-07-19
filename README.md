# 1. Introduction
I started this project for making NVIDIA Jetbot to play soccer game. I mainly use a Gazebo of ROS for training Jetbot to play soccer using Deep Reinforcmenet Learning. After training in virtual environment, trained model is copied to real world Jetbot. I want to check that kind of approach works well.

<img src="image/POM_Jetson.png"> <strong>I get a prize from NVIDIA for this project!</strong>

I can get a code of original Jetbot from https://github.com/dusty-nv/jetbot_ros/tree/master/gazebo.
All code are based on URDF official tutorial of ROS http://gazebosim.org/tutorials?tut=ros_urdf where I could learn how to make and simulate a robot in Gazebo. 
I will upload a detailed post to https://kimbring2.github.io/2019/10/26/jetbot.html. Please see it if you need more information about code. 

# 2. Python version
Currently, Gazebo only can be operated on Python 2.7. Thus, you should use a 2.7 version environment.

# 3. Python package
1. tensorflow-gpu==1.13.1 
2. opencv-python
3. cvlib(0.1.0), requests, progressbar, keras 

# 4. Reference
1. Jetbot SDF file, ROS : [Jetbot SDF file, ROS](https://github.com/dusty-nv/jetbot_ros)
2. Gazebo parameter setting: [Gazebo parameter](https://github.com/CentroEPiaggio/irobotcreate2ros)
3. URDF file usage in Gazebo : [URDF file usage in Gazebo](http://gazebosim.org/tutorials/?tut=ros_urdf)
4. Object detecion using cvlib: [Object detecion using cvlib](https://towardsdatascience.com/object-detection-with-less-than-10-lines-of-code-using-python-2d28eebc5b11)
5. Soccer field, ball model : [Soccer field, ball model](https://github.com/RoboCup-MSL/MSL-Simulator)
6. Reinforcement Learnig model : [Reinforcement Learnig model](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-6-partial-observability-and-deep-recurrent-q-68463e9aeefc)
7. Inference saved model : [Tensorrt](http://litaotju.github.io/2019/01/24/Tensorflow-Tutorial-6,-Using-TensorRT-to-speedup-inference/)
8. Onshape 3D model to URDF : [onshape-to-robot](https://github.com/rhoban/onshape-to-robot/)
9. GPIO control for solenoid electromagnet : https://www.jetsonhacks.com/2019/06/07/jetson-nano-gpio/ ,https://github.com/NVIDIA/jetson-gpio
10. Ball kicking mechanism : https://www.youtube.com/watch?v=fVGrYoqn-EU
11. How to read LaserScan data (ROS python) : https://www.theconstructsim.com/read-laserscan-data/

# 5. Simualtion, real part
The purpose of this project is to train Jetbot to play soccer based on simulation and then apply trained model to actual Jetbot. Therefore, I am currently updating the code and description of the current simulation robot and the actual robot to this repository together. However, you can run only simulation without any actual hardware.

# 6. How to build ROS project
At your terminal, run below command.

```
$ cd ~/catkin_ws/src/
$ git clone https://github.com/kimbring2/jetbot_gazebo.git
$ cd ..
$ catkin_make
$ source devel/setup.bash
```
# 7. Dependent ROS package install
Put a 'https://github.com/kimbring2/jetbot_soccer/tree/master/spawn_robot_tools' folder to your 'catkin_ws/src' folder.

# 8. How to view 3D model of Jetbot in Rviz
```
$ roslaunch jetbot_description jetbot_rviz.launch
```

# 9. How to start control Jetbot in roslaunch
```
roslaunch jetbot_gazebo main.launch
```

# 10. Soccer object model path setting
You should change a modeling path of jetbot/jetbot_gazebo/world/jetbot.world and sdf file at jetbot_gazebo/models/RoboCup15_MSL_Field, jetbot_gazebo/models/RoboCup15_MSL_Goal, jetbot_gazebo/models/football.

It is just example line of uri. Please change all uri path for your PC environment.
```
<uri>file:///home/[your ubuntu account]/catkin_ws/src/jetbot_soccer/jetbot_gazebo/materials/scripts/gazebo.material</uri>
```

# 11. Troubleshooting 
## RLException Error
If you get a 'RLException' error message, use 'source devel/setup.bash' command and try again.

<img src="image/Error_Message.png" width="600">

## 1) Could not find the GUI, install the 'joint_state_publisher_gui' package Error
If you get that error when try to run 'roslaunch jetbot_description jetbot_rviz.launch' command, use 'sudo apt install ros-melodic-joint-state-publisher-gui' command for installing it.

<img src="image/joint_state_error.jpg" width="600">

## 2) Could not load controller Error
If you get a 'Could not load controller' error message, try to install related package using below command at your terminal.

```
$ sudo apt-get install ros-melodic-ros-control ros-melodic-ros-controllers
```

<img src="image/controller_error.png" width="600">

# 12. How to manually send a wheel velocity commands
The range of velocity that can be given to the wheel is 0 to 100.

## 1) Left Wheel 
For robot1
```rostopic pub -1 /robot1/joint1_velocity_controller/command std_msgs/Float64 "data: 30"```

For robot2
```rostopic pub -1 /robot2/joint1_velocity_controller/command std_msgs/Float64 "data: 30"```

## 2) Right Wheel
For robot1
```rostopic pub -1 /robot1/joint2_velocity_controller/command std_msgs/Float64 "data: 30"```

For robot2
```rostopic pub -1 /robot2/joint2_velocity_controller/command std_msgs/Float64 "data: 30"```

# 13. Python code for Gazebo simulator
Move to 'jetbot/jetbot_control/src/' folder and type ```python main.py```. 
It will send a velocity command to each wheel and show a camera sensor image. Furthermore, Tensorflow code for Reinforcement Learning is implemented. Jetbot is able to only learn how to track a soccer ball at now. However, I train more advanced behavior after finishing first task.

If you run a code, it will store a Tensorflow weight file at drqn folder of your workspace. 

# 14. Python code for real Jetbot
First, set up ROS in actual Jetbot hardware based on manual of https://github.com/dusty-nv/jetbot_ros.

Then run roscore on Jetbot terminal and publish the camera frame using jetbot_camera node.
```
$ roscore 
$ rosrun jetbot_ros jetbot_camera 
```

You can control a wheel motor using below Python script. 
```
$ rosrun jetbot_ros jetbot_soccer_motors.py 
$ rostopic pub -1 /jetbot_soccer_motors/cmd_str std_msgs/String --once "30"
```

You can control a roller and solenoid motor using two Python script. 
```
$ rosrun jetbot_ros jetbot_soccer_roller.py 
$ rostopic pub -1 /jetbot_soccer_roller/cmd_str std_msgs/String --once "in"
$ rostopic pub -1 /jetbot_soccer_roller/cmd_str std_msgs/String --once "out"
```

```
$ rosrun jetbot_ros jetbot_soccer_solenoid.py 
$ rostopic pub -1 /jetbot_soccer_solenoid/cmd_str std_msgs/String --once "in"
$ rostopic pub -1 /jetbot_soccer_solenoid/cmd_str std_msgs/String --once "out"
```

You can also give a control command using Python code. Run 'jetbot_ros.py' file.
```$ python jetbot_ros.py ```

That file receive a image frame from camera and send a velecity command to each wheel. In this code, detecting soccer ball is performed using jetson.utils, jetson.inference.

<img src="image/jetbot_soccer_detect_ball.jpeg" width="600">

# 15. Tensorflow model freezing for TensorRT inference
Tensorflow model trained using Gazebo simulation can be used without installing Tensorflow on Jetson Nano. However, model needs to be freezed. Please check a process for it at 'RL_model_froze.ipynb' file. You need to change a 'model_dir = "/home/kimbring2/catkin_ws/src/jetbot/jetbot_control/src/drqn"' line for your workplace setting.

<img src="image/jetbot_frozen_graph.png" width="600">

You need to see a inference output at bottom of cell and modify 'model-1.cptk.meta' for your checkpoint name.

# 16. Modify Jetbot for soccer
I remodel hardware of Jetbot because it is not suitable for soccer. As you know easily, soccer robot needd a kicking and holding part. The Jetbot soccer version can hold a soccer ball and kick it. The wheel part is changed to omniwheel type for moving more freely. Battery, DC motor, WiFi antenna of previous Jetbot are reused for easy developing.

<img src="/image/jetbot_soccer_proto_2.png" width="600">
I use ancOnshape cloud 3D modeling to create a model. You can check and download my model from below link.

[Modified Jetbot 3D model Onshape link](https://cad.onshape.com/documents/242e5d0f2f1cbff393c8e507/w/37c9eecd4ded31866f99420c/e/9a6f236fb48a5317e2b639700)

[![Protoype test](https://img.youtube.com/vi/zNTldaCe1ZQ/0.jpg)](https://youtu.be/zNTldaCe1ZQ "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

After making 3D modeling, I convert it to URDF format for Gazebo simulation. I find and use a very convenient tool for that(https://github.com/rhoban/onshape-to-robot/)  

# 17. Dynamixel SDK test
The best way to use Dynamixel on Jetson Nano is using the SDK provided by ROBOTIS.

1. Check your connection between motor and control board(I use a Dynamixel Wizard for checking a operation of motor).
2. First, download a SDK from 'https://github.com/ROBOTIS-GIT/DynamixelSDK.git' to your Jetson Nano.
3. Move to 'DynamixelSDK/python/tests/protocol1_0' and run 'ping.py'.

<img src="/image/dynamixel_ping_test.png" width="600">

4. Open 'read_write.py' and change a parameter for MX-12W(You can also change the parameter using Dynamixel Wizard).
<img src="/image/rw_setting.png" width="600">

5. Run 'read_write.py' and you should see a success message like a below.
<img src="/image/rw_success.png" width="600">

[![Dynamixel test 2](https://img.youtube.com/vi/ZSii66zur4s/0.jpg)](https://youtu.be/ZSii66zur4s "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

# 18. RViz test for Jetbot soccer version
You can see a RViz 3D model of Jetbot soccer using below command.
```
roslaunch jetbot_description jetbot_soccer_rviz.launch
```

After launching a RViz, you can control of each wheel and roller using dialog box.

# 19. Gazebo test for Jetbot soccer version
After checking operation of each part at RViz, try to control it in Gazebo simulation.

```
roslaunch jetbot_gazebo main_soccer.launch
```

You can control of each wheel, roller, solenoid motor using 'rostopic pub' command.
First, adjust the speed of the wheels to approach to the ball.

1. Command for wheel motor
```
rostopic pub -1 /robot1/wheel1_velocity_controller/command std_msgs/Float64 "data: 30"
rostopic pub -1 /robot1/wheel2_velocity_controller/command std_msgs/Float64 "data: 30"
rostopic pub -1 /robot1/wheel3_velocity_controller/command std_msgs/Float64 "data: 30"
rostopic pub -1 /robot1/wheel4_velocity_controller/command std_msgs/Float64 "data: 30"
```

Next, rotate a roller motor to pull the ball.

2. Command for roller motor
```
rostopic pub -1 /robot1/roller_velocity_controller/command std_msgs/Float64 "data: 30"
```

Finally kick the ball via speed control of solenoid motor.

3. Command for solenoid motor
```
rostopic pub -1 /robot1/stick_velocity_controller/command std_msgs/Float64 "data: 30"
```

If you run a 'main_soccer.py file in jetbot/jetbot_control file, you can give a command by typing a character.
```
s : stop
f : forward
l : left
r : right
h : hold ball
k : kick ball
```

Please check video for checking how to give a command(https://www.youtube.com/watch?v=rTVKIcgdVGo)

# 20. Teleoperation test for real Jetbot soccer version
Like the original version of Jetbot, Jetbot soccer version can be controlled by gamepad. You can check a code for that teleoperation_soccer.ipynb file. Upload it to Jetson Nano and run it.

[![Teleoperation test](https://img.youtube.com/vi/vONoIruznlw/hqdefault.jpg)](https://www.youtube.com/watch?v=vONoIruznlw "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

You can use the gamepad for performing the basic actions for soccer. Multi players will be able to play robot soccer together if power of robot is a bit more reinforced. It is little weak for playing real soccer.

# 21. Citation
If you use DeepSoccer to conduct research, we ask that you cite the following paper as a reference:

<blockquote>
<cite>
@misc{kim2020deepsoccer,
  author = {Dohyeong, Kim},
  title = {DeepSoccer},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/kimbring2/DeepSoccer/}},
  commit = {5689b3ab8934bc2f60360e3b180978b637fb2741}
}
</cite>
</blockquote>

# 22. License
Apache License 2.0
