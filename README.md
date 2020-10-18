# 1. Introduction
The purpose of this project is making a soccer robot. For this purpose, various methods and tools are intoduced such as Robot Operation System (ROS) for robot control, Deep Reinforcement Learning for control algorithm. 

Due to the characteristic of Deep Learning, a large amount of training data is required. Thus, virtual simulation tool of ROS(Gazebo) is additionally used. The project uses the basic Deep Reinforcement Learning training and evaluation process. 

In order to use the robot algorithm trained in the virtual simulation in the real world, technique for reducing gap between simulation and real world such as domain randomization will be used.

In addition to opening software, information about hardware of robot will be shared for making other reseachres, makers can use this project for their own purpose.

[![Project Introduction](https://img.youtube.com/vi/BQWncZ6QNDE/hqdefault.jpg)](https://youtu.be/BQWncZ6QNDE "Jetbot Soccer Play - Click to Watch!")
<strong>Project Introduction!</strong>

More detailed instruction can be found at my [blog post of DeepSoccer](https://kimbring2.github.io/2020/10/08/deepsoccer.html)

# 2. Software Dependency
## 1) ROS, Gazebo
- ROS Melodic, Gazebo 9
- ROS openai_ros package
- Gazebo only provides Python 2.7(From ROS Noetic at Ubuntu 20.04, Python3 can be used)

## 2) Python 
- Tensorflow 2.1.0
- requests 
- pynput
- progressbar
- opencv-python

## 3) Jetson Nano
- JetPack 4.4
- ROS Melodic
- Tensorflow 2.2.0
- Python2 for ROS acuator, sensor node
- Python3 for ROS main node

# 3. Reference
- Jetbot SDF file, ROS: [Jetbot SDF file, ROS](https://github.com/dusty-nv/jetbot_ros)
- Gazebo parameter setting: [Gazebo parameter](https://github.com/CentroEPiaggio/irobotcreate2ros)
- URDF file usage in Gazebo: [URDF file usage in Gazebo](http://gazebosim.org/tutorials/?tut=ros_urdf)
- Object detecion using cvlib: [Object detecion using cvlib](https://towardsdatascience.com/object-detection-with-less-than-10-lines-of-code-using-python-2d28eebc5b11)
- Soccer field, ball model: [Soccer field, ball model](https://github.com/RoboCup-MSL/MSL-Simulator)
- Reinforcement Learnig model: [Reinforcement Learnig model](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-6-partial-observability-and-deep-recurrent-q-68463e9aeefc)
- Inference saved model: [Tensorrt](http://litaotju.github.io/2019/01/24/Tensorflow-Tutorial-6,-Using-TensorRT-to-speedup-inference/)
- Onshape 3D model to URDF: [onshape-to-robot](https://github.com/rhoban/onshape-to-robot/)
- GPIO control for solenoid electromagnet : https://www.jetsonhacks.com/2019/06/07/jetson-nano-gpio/ ,https://github.com/NVIDIA/jetson-gpio
- Ball kicking mechanism: https://www.youtube.com/watch?v=fVGrYoqn-EU
- How to read LaserScan data(ROS python): https://www.theconstructsim.com/read-laserscan-data/
- Convert Video to Images (Frames) & Images (Frames) to Video using OpenCV (Python) : https://medium.com/@iKhushPatel/convert-video-to-images-images-to-video-using-opencv-python-db27a128a481
- Python Multithreading with pynput.keyboard.listener: https://stackoverflow.com/a/59520236/6152392
- How to use a Gazebo as type of OpenAI Gym: http://wiki.ros.org/openai_ros
- Solenoid joint spring plugin: https://github.com/aminsung/gazebo_joint_torsional_spring_plugin
- Custom control plugin for Gazebo: http://gazebosim.org/tutorials?tut=guided_i5&cat=
- Forgetful Expirience Replay for Reinforcement Learning from Demonstrations: https://github.com/cog-isa/forger
- Compiling ROS cv_bridge with Python3: https://cyaninfinite.com/ros-cv-bridge-with-python-3/
- Style Transfer for Sim2Real: https://github.com/cryu854/FastStyle
- CycleGAN for Sim2Real: https://www.tensorflow.org/tutorials/generative/cyclegan
- Image Segmentation for CycleGAN: https://www.kaggle.com/santhalnr/cityscapes-image-segmentation-pspnet

# 4. Etc
## 1) Relationship between simualtion and real part
The purpose of this project is to train Jetbot to play soccer based on simulation and then apply trained model to actual Jetbot. Therefore, I am currently updating the code and description of the current simulation robot and the actual robot to this repository together. However, you can run only simulation without any actual hardware.

## 2) How to build ROS project
At your terminal, run below command.

```
$ cd ~/catkin_ws/src/
$ git clone https://github.com/kimbring2/DeepSoccer.git
$ cd ..
$ catkin_make
$ source devel/setup.bash
```

## 3) Dependent ROS package install
Put a 'https://github.com/kimbring2/DeepSoccer/tree/master/spawn_robot_tools' folder to your 'catkin_ws/src' folder.

# 5. Troubleshooting 
## 1) RLException Error
If you get a 'RLException' error message, use 'source devel/setup.bash' command and try again.

## 2) Could not find the GUI, install the 'joint_state_publisher_gui' package Error
If you get that error when try to run 'roslaunch jetbot_description jetbot_rviz.launch' command, try to install related package using below command at your terminal.

```
$ sudo apt install ros-melodic-joint-state-publisher-gui
```

## 3) Could not load controller Error
If you get a 'Could not load controller' error message, try to install related package using below command at your terminal.

```
$ sudo apt-get install ros-melodic-ros-control ros-melodic-ros-controllers
```

## 4) RViz 'No transform from' error

If you get error message includes 'No transform from', try to install unicode ubuntu package and reboot.

```
$ sudo apt-get install unicode 
```

## 5) Python code for real robot
First, set up ROS in actual Jetbot hardware based on manual of https://github.com/dusty-nv/jetbot_ros.

Then run roscore on Jetbot terminal and publish the camera frame using jetbot_camera node.
```
$ roscore 
$ rosrun jetbot_ros jetbot_camera 
```

You can control a wheel motor using below Python script. 
```
$ rosrun jetbot_ros jetbot_soccer_motors.py 
$ rostopic pub -1 /deepsoccer_motors/cmd_str_wheel1 std_msgs/String --once "'30'"
$ rostopic pub -1 /deepsoccer_motors/cmd_str_wheel2 std_msgs/String --once "'30'"
$ rostopic pub -1 /deepsoccer_motors/cmd_str_wheel3 std_msgs/String --once "'30'"
$ rostopic pub -1 /deepsoccer_motors/cmd_str_wheel4 std_msgs/String --once "'30'"
```

You can control a roller and solenoid motor using two Python script. 
```
$ rosrun jetbot_ros jetbot_soccer_roller.py 
$ rostopic pub -1 /deepsoccer_roller/cmd_str std_msgs/String --once "in"
$ rostopic pub -1 /deepsoccer_roller/cmd_str std_msgs/String --once "out"
```

```
$ rosrun jetbot_ros jetbot_soccer_solenoid.py 
$ rostopic pub -1 /deepsoccer_solenoid/cmd_str std_msgs/String --once "in"
$ rostopic pub -1 /deepsoccer_solenoid/cmd_str std_msgs/String --once "out"
```

For getting lidar sensor distance and infrared object detection value.
```
$ sudo chmod a+rw /dev/ttyTHS1 
$ rosrun jetbot_ros jetbot_soccer_lidar.py
$ rostopic echo /deepsoccer_lidar
```

```
$ rosrun jetbot_ros jetbot_soccer_infrared.py
$ rostopic echo /deepsoccer_infrared
```

You can also give a control command using Python code. Run 'jetson_soccer_main.py' file at Jetson Nano terminal.
```$ python jetson_soccer_main.py ```

# 6. DeepSoccer design
I remodel hardware of Jetbot because it is not suitable for soccer. As you know easily, soccer robot needd a kicking and holding part. The Jetbot soccer version can hold a soccer ball and kick it. The wheel part is changed to omniwheel type for moving more freely. Battery, DC motor, WiFi antenna of previous Jetbot are reused for easy developing.

<img src="/image/jetbot_soccer_proto_2.png" width="600">
I use Onshape cloud 3D modeling program to create a model. You can check and download my model from below link.

[Modified Jetbot 3D model Onshape link](https://cad.onshape.com/documents/242e5d0f2f1cbff393c8e507/w/37c9eecd4ded31866f99420c/e/9a6f236fb48a5317e2b639700)

After making 3D modeling, I convert it to URDF format for Gazebo simulation. I find and use a very convenient tool for that(https://github.com/rhoban/onshape-to-robot/)  

## 1) Dynamixel SDK test
The best way to use Dynamixel on Jetson Nano is using the SDK provided by ROBOTIS.

- Check your connection between motor and control board(I use a Dynamixel Wizard for checking a operation of motor).
- First, download a SDK from 'https://github.com/ROBOTIS-GIT/DynamixelSDK.git' to your Jetson Nano.
- Move to 'DynamixelSDK/python/tests/protocol1_0' and run 'ping.py'.

<img src="/image/dynamixel_ping_test.png" width="600">

- Open 'read_write.py' and change a parameter for MX-12W(You can also change the parameter using Dynamixel Wizard).
<img src="/image/rw_setting.png" width="600">

- Run 'read_write.py' and you should see a success message like a below.
<img src="/image/rw_success.png" width="600">

[![Dynamixel test 2](https://img.youtube.com/vi/ZSii66zur4s/0.jpg)](https://youtu.be/ZSii66zur4s "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

## 2) RViz test
You can see a RViz 3D model of Jetbot soccer using below command.
```
$ roslaunch deepsoccer_description deepsoccer_rviz.launch
```

After launching a RViz, you can control of each wheel and roller using dialog box.

## 3) Gazebo test
After checking operation of each part at RViz, try to control it in Gazebo simulation.

```
$ roslaunch deepsoccer_gazebo main_soccer.launch
```

You can control of each wheel, roller, solenoid motor using 'rostopic pub' command.
First, adjust the speed of the wheels to approach to the ball.

- Command for wheel motor
```
$ rostopic pub -1 /robot1/wheel1_velocity_controller/command std_msgs/Float64 "data: 30"
$ rostopic pub -1 /robot1/wheel2_velocity_controller/command std_msgs/Float64 "data: 30"
$ rostopic pub -1 /robot1/wheel3_velocity_controller/command std_msgs/Float64 "data: 30"
$ rostopic pub -1 /robot1/wheel4_velocity_controller/command std_msgs/Float64 "data: 30"
```

Next, rotate a roller motor to pull the ball.

- Command for roller motor
```
$ rostopic pub -1 /robot1/roller_velocity_controller/command std_msgs/Float64 "data: 30"
```

Finally, kick the ball via speed control of solenoid motor.

- Command for solenoid motor
```
$ rostopic pub -1 /robot1/stick_velocity_controller/command std_msgs/Float64 "data: 30"
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

## 4) Command for lidar sensor
Soccer robot need to check a obstacle of front side. Using only camera sensor is not enough for that. Thus, I decide adding lidar sensor. Information of lidar sensor can be checked by using ROS topic named '/deepsoccer/laser/scan'

```
$ rostopic echo /deepsoccer/laser/scan -n1
```

Among that information, range from robot to front object can be got by using Python
```
def lidar_callback(msg):
    global lidar_range

    lidar_range = msg.ranges[360]
```

Gazebo simulator visualize the range of the lidar sensor. You can see the range value of lidar sensor is changed depending on the distance between the robot and front obstacle.

[![Jetbot soccer lidar sensor simulation test](http://i3.ytimg.com/vi/2b6BUH5tF1g/hqdefault.jpg)](https://youtu.be/2b6BUH5tF1g "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

## 5) Teleoperation test
Like the original version of Jetbot, Jetbot soccer version can be controlled by gamepad. You can check a code for that teleoperation_soccer.ipynb file. Upload it to Jetson Nano and run it.

[![Teleoperation test](https://img.youtube.com/vi/vONoIruznlw/hqdefault.jpg)](https://www.youtube.com/watch?v=vONoIruznlw "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

You can use the gamepad for performing the basic actions for soccer. Multi players will be able to play robot soccer together if power of robot is a bit more reinforced. It is little weak for playing real soccer.

## 6) Gazebo solenoid electromagnet joint plugin
Since the jetbot soccer version uses solenoid electromagnet for kicking ball which has a spring, so it cannot be implemented using default controller of Gazebo. In such a case, we are able to create a custom plugin. First, 'solenoid_electromagnet_joint_spring_plugin' package need be built using 'catkin_make' command.

<img src="/image/Spring-Constant.jpg" width="600">

```
<gazebo>
   <!-- joint torsional spring plugin -->
   <plugin name="stick_solenoid_electromagnet_joint_spring" filename="libsolenoid_electromagnet_joint_spring_plugin.so">
     <kx>1000000000</kx>
     <set_point>0.0</set_point>
     <joint>stick</joint>
   </plugin>
</gazebo>
```

The built custom plugin is used for sticks in multiple joints, so you can declare it in the jetbot_soccer.gazebo file as above.

## 7) Use DeepSoccer as OpenAI Gym format 
Most Deep Reinforcement Learning researchers are accustomed to Gym environment of OpenAI. There is package called openai_ros that allows user use a custom robot environment in the form of Gym. 

DeepSoccer also provides a package for use a it as Gym format. First, download a my_deepsoccer_training pacakge from this repo. After that, copy it to the src folder under ROS workspace like a Jetbot package and build it.

The my_deepsoccer_training package is based on the my_turtlebot2_training package from the http://wiki.ros.org/openai_ros tutorial. I recommend that you first run a  tutorial package successfully.

After installing the my_deepsoccer_training package, you can use DeepSoccer with the following Gym shape. The basic actions and observations are the same as described in the Jetbot soccer section. Action is an integer from 0 to 6, indicating STOP, FORWARD, LEFT, RIGHT, BACKWARD, HOLD, and KICK, respectively. Observations are image frame from camera, robot coordinates, and lidar sensor value.

After changing a line of start_training.launch like below.

```
<node pkg="my_deepsoccer_training" name="deepsoccer_single" type="gym_test.py" output="screen"/>
```

You can check code for it at [gym_test.py file](https://github.com/kimbring2/DeepSoccer/blob/master/my_deepsoccer_training/src/gym_test.py).

Start Gazebo by using below command.
```
$ roslaunch my_deepsoccer_training start_training.launch
```

## 8) Collect your playing dataset
Since Reinforcement Learning used in DeepSoccer is a method that uses expert data, user can control a robot directly. For using [collecting_human_dataset.py file](https://github.com/kimbring2/DeepSoccer/blob/master/my_deepsoccer_training/src/collecting_human_dataset.py) for that, you need to change a line of launch file located in [launch folder](https://github.com/kimbring2/DeepSoccer/tree/master/my_deepsoccer_training/launch).

```
<node pkg="my_deepsoccer_training" name="deepsoccer_single" type="collecting_human_dataset.py" output="screen"/>
```

After that, launch a Gazebo using below command.
```
$ roslaunch my_deepsoccer_training start_training.launch
```

Once Gazebo is started, you can give commands to the robot using the keyboard keys. S is stop, f is forward, l is left, r is right, b is reverse, h is catching the soccer ball, k is kicking ball, and p is the running. When you press the q key, the recorded data is saved in the folder and the entire program ends.

You can set the path and name of saving file by changing a save_path and save_file options of [my_deepsoccer_single_params.yaml file](https://github.com/kimbring2/DeepSoccer/blob/master/my_deepsoccer_training/config/my_deepsoccer_single_params.yaml).


## 9) Training DeepSoccer using Deep Reinforcement Learning
After making DeepSoccer in Openai Gym format, let's use it trarning robot using Deep Reinforcement Learning. Currently, the most commonly used Deep Reinforcement Learning algorithms like PPO are good when the action of the agent is relatively simple. However, DeepSoccer agent has to deal with soccer ball very delicately. Thus, I assume that PPO alorithm do not work well in this project. For that reason, I decide to use a one of Deep Reinforcement Learning method "Forgetful Experience Replay in Hierarchical Reinforcement Learning from Demonstrations", which operates in the complex environment like a soccer, by mixing trained agent data and expert demonstration data.

The code related to this algorithm is be located at [ForgER folder](https://github.com/kimbring2/DeepSoccer/tree/master/my_deepsoccer_training/src/ForgER). 

You can train a robot using human demonstration data(https://drive.google.com/drive/folders/18kqrpbLMGEnAOd1QTHCRzL_VyUCGItcE?usp=sharing). Change a line of launch file like that.

```
<node pkg="my_deepsoccer_training" name="deepsoccer_single" type="train_single.py" output="screen"/>
```

File for this script is located at [train_single.py file](https://github.com/kimbring2/DeepSoccer/blob/master/my_deepsoccer_training/src/train_single.py). There are four important function in that file. 

```
agent.add_demo()
agent.pre_train(config['pretrain']['steps'])
agent.train(env, name="model.ckpt", episodes=config['episodes'])
agent.test(env)
```

Human demonstration data is added to buffer by add_demo function. Next, agent is trained using pre_train function. After finishing pretraining, agent can be trained using a data from interaction between agent and environemnt using train() function. Every trained result is able to be checked by using test(function).

Start Gazebo by using below command.
```
$ roslaunch my_deepsoccer_training start_training.launch
```

All parameters related to Reinforcmeent Learning can be checked at [deepsoccer_config.yaml file](https://github.com/kimbring2/DeepSoccer/blob/master/my_deepsoccer_training/src/deepsoccer_config.yaml). Buffer size and pretrain steps are important. Save_dir, tb_dir parameter means saving location of trained Tensorflow model and Tensorboard log file.   

## 10) Using pretrained model at Jetson Nano 
In order to use the model trained by Gazebo simulation at Jetson Nano. You need to copy a folder named pre_trained_model.ckpt generated after training at previous step. Inside the folder, there are assets and variables folders, and frozen model named saved_model.pb.

After placing [jetbot_ros folder](https://github.com/kimbring2/DeepSoccer/tree/master/jetbot_ros) to your ROS workspace of Jetson Nano, run below command.

```
$ roscore
$ roslaunch jetbot_ros start.launch
```

It will launch all actuator and sensor ROS node. After that, change a pre_trained_model.ckpt folder path what you copied at [jetbot_soccer_main.py](https://github.com/kimbring2/DeepSoccer/blob/master/jetbot_ros/scripts/jetbot_soccer_main.py). Next, move to script folder of jetbot_ros ROS package and run below command.

```
$ python3 jetbot_soccer_main.py
```

Because Tensorflow 2 of Jetson Nano only can be run by Python3, you need to do one more job because cv_bridge of ROS melodic is not able to be ran at Python3. Please follow a intruction at https://cyaninfinite.com/ros-cv-bridge-with-python-3/.

If the tasks described on the above site are completed successfully, DeepSoccer start to control acuator based on the data from real sensor.

[![Jetbot soccer Deep Reinforcement Learning training result](https://img.youtube.com/vi/Ur7L5j9fIwY/sddefault.jpg)](https://youtu.be/Ur7L5j9fIwY "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

It is confirmed that robot do not show the same movement as the trained one when the raw camera frame is used as input to the RL model.

# 7. Sim2Real method
## 1) Concept intruction
Unlike humans, robots cannot respond appropriately to environment that is different from the simulation environment. Therefore, the real world information must be converted to the simulation environment. Recently, there are several ways to apply deep learning to these Sim2Real. One of method is using Neural Style Transfer and another is applying CycleGAN. I apply both of methods to DeepSoccer and check it is working properly.

<img src="/image/sim2real_concept.png" width="600">

## 2) Neural Style Transfer approach
I use a code of https://github.com/cryu854/FastStyle for Neural Style Transfer. The advantage of this method is that you only need one conversion target style image without collecting train images separately, but this method does not completely convert to simulation.

<img src="/image/raw-video.gif" width="530"> <img src="/image/styled-video.gif" width="300">

You can train your own model using code of that repo and real world image. Altenatively, you can also use the [pretrained model](https://drive.google.com/drive/folders/1_JL-JK7uDjNfkDlSBvTzubzGzU5Vj51L?usp=sharing) of the DeepSoccer Gazebo simulation image.

```
import cv2
import numpy as np
import tensorflow as tf

imported_style = tf.saved_model.load("/home/[your Jetson Nano user name]/style_model")
f_style = imported_style.signatures["serving_default"]
style_test_input = np.zeros([1,256,256,3])
style_test_tensor = tf.convert_to_tensor(style_test_input, dtype=tf.float32)
f_style(style_test_tensor)['output_1']

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
if cap.isOpened() != 1:
    continue

ret, frame = cap.read()
img = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
            
img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
resized = np.array([img])
input_tensor = tf.convert_to_tensor(resized, dtype=tf.float32)
output_style = f_style(input_tensor)['output_1'].numpy()
cv2.imwrite("output_style.jpg", output_style)
```

You can save the pretrain model to your Jetson Nano and use the above code to try to run Neural Style Transfer.

## 3) CycleGAN approach
The method using CycleGAN is training a model by dataset of real and simulation world. For this method, I refer to the [method of official Tensorflow website](https://www.tensorflow.org/tutorials/generative/cyclegan).

As can be seen in the [real world dataset](https://drive.google.com/drive/folders/1TuaYWI191L0lc4EaDm23olSsToEQRHYY?usp=sharing), there are many objects in the background of the experimental site such as chair, and umbrella. If I train the CycleGAN model with the [simulation world dataset](https://drive.google.com/drive/folders/166qiiv2Wx0d6-DZBwHiI7Xgg6r_9gmfy?usp=sharing) without removing background objects, I am able to see the problem of the chair turning into goalpost.

<img src="/image/CycleGAN_wrong_case_4.png" width="400"> <img src="/image/CycleGAN_wrong_case_7.png" width="400">

In order to solve this problem, I first decide that it is necessary to delete all objects except the goal, goalpost, and floor that the robot should recognize to play soccer. Segmentation using classic OpenCV method do not work. On the other hand, Deep Learning model using the [ADE20K dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/) can segregate object well. You can check [code for segmentation](https://github.com/kimbring2/DeepSoccer/blob/master/segmentation.ipynb). Robot do not have to separate all the object in the dataset. Thus, I modify the ADE20K dataset a bit like a below.

<img src="/image/ADE_train_00006856.jpg" width="300"> <img src="/image/ADE_train_00006856_seg.png" width="300"> <img src="/image/ADE_train_00006856_seg_simple.png" width="300">

[![Jetbot soccer lidar sensor simulation test](https://img.youtube.com/vi/a5IjHdsv_eA/0.jpg)](https://youtu.be/a5IjHdsv_eA "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

# 8. Citation
If you use DeepSoccer to conduct research, we ask that you cite the following paper as a reference:

```
@misc{kim2020deepsoccer,
  author = {Dohyeong, Kim},
  title = {DeepSoccer},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/kimbring2/DeepSoccer/}},
  commit = {9ccab28a7e2a9a14caa119a765f95e2c6d0b044e}
}
```

# 9. Acknowledgement
<img src="image/POM_Jetson.png"> <strong>I get a prize from NVIDIA for this project</strong>

# 10. License
Apache License 2.0
