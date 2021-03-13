# 1. Software Dependency
- ROS Melodic, Gazebo 9
- ROS openai_ros package
- Gazebo only provides Python 2.7(From ROS Noetic at Ubuntu 20.04, Python3 can be used)
- Tensorflow 2.1.0
- requests 
- pynput
- progressbar
- opencv-python

# 2. Usage
At your terminal, run below command.

```
$ cd ~/catkin_ws/src/
$ git clone https://github.com/kimbring2/DeepSoccer.git
$ cd ..
$ catkin_make
$ source devel/setup.bash
```

# 3. Troubleshooting 
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

# 4. DeepSoccer design
I remodel hardware of Jetbot because it is not suitable for soccer. As you know easily, soccer robot needd a kicking and holding part. The Jetbot soccer version can hold a soccer ball and kick it. The wheel part is changed to omniwheel type for moving more freely. Battery, DC motor, WiFi antenna of previous Jetbot are reused for easy developing.

I use Onshape cloud 3D modeling program to create a model. You can see [DeepSoccer 3D model](https://cad.onshape.com/documents/242e5d0f2f1cbff393c8e507/w/37c9eecd4ded31866f99420c/e/9a6f236fb48a5317e2b639700).

After making 3D model, I convert it to URDF format for Gazebo simulation using [onshape-to-robot](https://github.com/rhoban/onshape-to-robot/).

# 5. Unit test
## 1) RViz test
You can see a RViz 3D model of Jetbot soccer using below command.
```
$ roslaunch deepsoccer_description deepsoccer_rviz.launch
```

After launching a RViz, you can control of each wheel and roller using dialog box.

## 2) Gazebo test
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

## 3) Command for lidar sensor
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

[![DeepSoccer lidar sensor simulation test](http://i3.ytimg.com/vi/2b6BUH5tF1g/hqdefault.jpg)](https://youtu.be/2b6BUH5tF1g "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

## 4) Teleoperation test
Like the original version of Jetbot, Jetbot soccer version can be controlled by gamepad. You can check a code for that teleoperation_soccer.ipynb file. Upload it to Jetson Nano and run it.

[![DeepSoccer teleoperation test](https://img.youtube.com/vi/vONoIruznlw/hqdefault.jpg)](https://www.youtube.com/watch?v=vONoIruznlw "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

You can use the gamepad for performing the basic actions for soccer. Multi players will be able to play robot soccer together if power of robot is a bit more reinforced. It is little weak for playing real soccer.

## 5) Gazebo solenoid electromagnet joint plugin
Since the jetbot soccer version uses solenoid electromagnet for kicking ball which has a spring, it cannot be implemented using default controller of Gazebo. In such a case, we are able to create a custom plugin. First, 'solenoid_electromagnet_joint_spring_plugin' package need be built using 'catkin_make' command.

<center><strong>Spring equation for solenoid electromagnet </strong></center>

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

The built custom plugin is used for stick joint. You need to declare it in the jetbot_soccer.gazebo file as like above.

# 6. Training robot in simulaation
## 1) Use DeepSoccer as OpenAI Gym format 
After changing a line of start_training.launch like below.

```
<node pkg="my_deepsoccer_training" name="deepsoccer_single" type="gym_test.py" output="screen"/>
```

You can check code for it at [gym_test.py file](https://github.com/kimbring2/DeepSoccer/blob/master/my_deepsoccer_training/src/gym_test.py).

Start Gazebo by using below command.
```
$ roslaunch my_deepsoccer_training start_training.launch
```

## 2) Collect your playing dataset
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

## 3) Training DeepSoccer using Deep Reinforcement Learning
You can train a robot using human demonstration data(https://drive.google.com/drive/folders/1s6hmnXj9IfdfTJzRg9rVuYyJ65MTTluU?usp=sharing). Change a line of launch file like that.

Robot soccer is quite difficult to hold the ball accurately unlike soccer games where you can control a ball by just pressing a key. The process for the robot to hold the ball is divided into several stages. First stage is rotating angle of robot for capturing ball inside of front camera angle. Next, robot must move to the ball keeping it in the center of the camera in order to hold it by the roller. Finally, robot should move forward towards the goal post while fixing the ball to the roller. When robot get close enough to the goal post, it can kick the ball using solenoid electromagnet.

<img src="/image/deepsoccer_reward.png" width="800">

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
