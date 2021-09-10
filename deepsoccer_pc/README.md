# 1. Software Dependency
- ROS Melodic, Gazebo 9
- ROS openai_ros package
- Gazebo only provides Python 2.7(From ROS Noetic at Ubuntu 20.04, Python3 can be used)
- Tensorflow 2.1.0
- requests 
- pynput
- progressbar
- opencv-python

# 2. Network Architecture
<img src="/image/network_architecture_sim.png" width="800">

# 3. Usage
## Set model file for soccer field
First, Copy [deepsoccer_gazebo](https://github.com/kimbring2/DeepSoccer/tree/master/deepsoccer_pc/deepsoccer_gazebo) folder to '/home/[Your User Name]/.gazebo/models' for soccer field rendering.

## Training network using Supervised Learning
After setting everything mentioned above, download the [human expert data](https://drive.google.com/drive/folders/1QmYI_FL5cym3LTvm8hlLfZ1Bo50bUyfc?usp=sharing) from Google Drive. I collected that data manually. 

After that, extract it to your workspace folder. You need to set a workspace_path in [yaml file](https://github.com/kimbring2/DeepSoccer/blob/master/my_deepsoccer_training/config/deepsoccer_params.yaml).

After preparing the human expert data and config file, run below command. It will start a [tutorial file](https://github.com/kimbring2/DeepSoccer/blob/master/my_deepsoccer_training/src/gym_tutorial.py) of DeepSoccer as format of OpenAI Gym environment. 

```
$ cd ~/catkin_ws/src/
$ git clone https://github.com/kimbring2/DeepSoccer.git
$ cd ..
$ catkin_make
$ source devel/setup.bash
$ roslaunch my_deepsoccer_training start_tutorial.launch
```

Trained model, Tensorboard log are saved under 'src/train/' folder seperately. Try to check the loss is decreased under 0.3 point. 

[![DeepSoccer Supervised Learning demo](https://img.youtube.com/vi/64ENY3P8U88/sddefault.jpg)](https://youtube.com/watch?v=64ENY3P8U88 "DeepSoccer- Click to Watch!")
<strong>Click to Watch!</strong>

## Training network using Reinforcement Learning
If you set the sl_training config to False and run the start_tutorial.launch as like above, it will start the Reinforcmenet Learning phase using trained model.

## Collection human expert data
<img src="/image/key_action_table.png" width="400">

You can also collect the human expert dataset using 'roslaunch my_deepsoccer_training start_dataset.launch' command. After Gazebo screen is poped up, you can control robot using 8 command.

<img src="/image/loss_sl.png" width="400">

The name of the replay file and the path to be saved can be set in the [config file](https://github.com/kimbring2/DeepSoccer/blob/master/my_deepsoccer_training/config/deepsoccer_params.yaml).

# 4. Troubleshooting 
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

# 5. Docker 
You can also use a Docker image for this repo using below command. Afte that, you can connect to created container by VNC(http://127.0.0.1:6080/#/).

```
$ docker run -it --rm -p 6080:80 --name=env_1 kimbring2/deepsoccer:latest
```

Because of low speed of Gazebo simulation in Docker image, you need to see Gazebo of it by using host PC. However, you shoud place ROS package to ' /root/catkin_ws/src' for rendering robot model. After that, get IP address of Docker image and connect to it by using below command.

```
$ export GAZEBO_MASTER_IP=$(docker inspect --format '{{ .NetworkSettings.IPAddress }}' env_1)
$ export GAZEBO_MASTER_URI=$GAZEBO_MASTER_IP:11345
$ gzclient --verbose
```

After connecting to the docker image with VNC, move to the catkin_ws folder. Then, you can start the simulation by executing the following command.
