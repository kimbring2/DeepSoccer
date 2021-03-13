# 1. Software Dependency
- JetPack 4.4
- ROS Melodic
- Tensorflow 2.2.0
- Python2 for ROS acuator, sensor node
- Python3 for ROS main node

# 2. Usage
First, set up ROS in actual Jetbot hardware based on manual of https://github.com/dusty-nv/jetbot_ros.

Then run roscore on Jetbot terminal and publish the camera frame using jetbot_camera node.
```
$ roscore 
$ rosrun deepsoccer_ros deepsoccer_camera 
```

You can control a wheel motor using below Python script. 
```
$ rosrun deepsoccer_ros deepsoccer_motors.py 
$ rostopic pub -1 /deepsoccer_motors/cmd_str_wheel1 std_msgs/String --once "'30'"
$ rostopic pub -1 /deepsoccer_motors/cmd_str_wheel2 std_msgs/String --once "'30'"
$ rostopic pub -1 /deepsoccer_motors/cmd_str_wheel3 std_msgs/String --once "'30'"
$ rostopic pub -1 /deepsoccer_motors/cmd_str_wheel4 std_msgs/String --once "'30'"
```

You can control a roller and solenoid motor using two Python script. 
```
$ rosrun deepsoccer_ros deepsoccer_roller.py 
$ rostopic pub -1 /deepsoccer_roller/cmd_str std_msgs/String --once "in"
$ rostopic pub -1 /deepsoccer_roller/cmd_str std_msgs/String --once "out"
```

```
$ rosrun deepsoccer_ros deepsoccer_solenoid.py 
$ rostopic pub -1 /deepsoccer_solenoid/cmd_str std_msgs/String --once "in"
$ rostopic pub -1 /deepsoccer_solenoid/cmd_str std_msgs/String --once "out"
```

For getting lidar sensor distance and infrared object detection value.
```
$ sudo chmod a+rw /dev/ttyTHS1 
$ rosrun deepsoccer_ros deepsoccer_lidar.py
$ rostopic echo /deepsoccer_lidar
```

```
$ rosrun deepsoccer_ros deepsoccer_infrared.py
$ rostopic echo /deepsoccer_infrared
```

You can start all node by just one command line.

```
$ roslaunch deepsoccer_ros start.launch
```

You can also give a control command using Python code. Run 'jetson_soccer_main.py' file at Jetson Nano terminal.
```$ python deepsoccer_main.py ```

# 3. Using pretrained model at Jetson board 
In order to use the model trained by Gazebo simulation at Jetson embedded board. You need to copy a folder named pre_trained_model.ckpt generated after training at previous step. Inside the folder, there are assets and variables folders, and frozen model named saved_model.pb.

After placing [DeepSoccer_ROS Package](https://github.com/kimbring2/DeepSoccer/tree/master/deepsoccer_jetson) to your ROS workspace of Jetson Xavier NX, run below command.

```
$ roscore
$ roslaunch deepsoccer_ros start.launch
```

It will launch all actuator and sensor ROS node. After that, change a pre_trained_model.ckpt folder path what you copied at [deepsoccer_main.py](https://github.com/kimbring2/DeepSoccer/blob/master/deepsoccer_jetson/scripts/deepsoccer_main.py). Next, move to script folder of deepsoccer_ros ROS package and run below command.

```
$ python3 deepsoccer_main.py
```

Because Tensorflow 2 of Jetson Xavier NX only can be run by Python3, you need to do one more job because cv_bridge of ROS melodic is not able to be ran at Python3. Please follow a intruction at https://cyaninfinite.com/ros-cv-bridge-with-python-3/.

If the tasks described on the above site are completed successfully, DeepSoccer start to control acuator based on the data from real sensor.

[![Deepsoccer Deep Reinforcement Learning training result](https://img.youtube.com/vi/Ur7L5j9fIwY/sddefault.jpg)](https://youtu.be/Ur7L5j9fIwY "DeepSoccer Play - Click to Watch!")
<strong>Click to Watch!</strong>
