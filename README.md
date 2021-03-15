# 1. Introduction
The purpose of this project is making a soccer robot. For this purpose, various methods and tools are introduced such as Robot Operation System (ROS) for robot control, and Deep Reinforcement Learning for controlling algorithm. 

Due to the characteristic of Deep Learning, a large amount of training data is required. Thus, virtual simulation tool of ROS called Gazebo is additionally used. The project uses the one of famous Deep Reinforcement Learning algorithm which uses a human expert data for improving performance.

In order to use the robot algorithm trained in the virtual simulation in the real world, I use a CycleGAN for generating view of simulation world from view of real world. 

Finally, hardware information of robot also will be shared as cad format for making other researchers, makers to use this project for their own purpose.

More detailed instruction can be found at my [blog post of DeepSoccer](https://kimbring2.github.io/2020/10/08/deepsoccer.html)

# 2. Robot design
I remodel hardware of Jetbot because it is not suitable for soccer. As you know easily, soccer robot needd a kicking and holding part. The Jetbot soccer version can hold a soccer ball and kick it. The wheel part is changed to omniwheel type for moving more freely.

<img src="image/DeepSoccer_hardware_design.png" width="450"> <img src="image/deepsoccer_hardware_v2_1.jpg" width="450">

You can see detailed information about hardware design at https://kimbring2.github.io/2020/10/08/deepsoccer.html#design_deepsoccer.

<img src="image/sim2real_instruction.png" width="800">

# 3. Relationship between simualtion and real part
The purpose of this project is to train Jetbot to play soccer based on simulation and then apply trained model to actual Jetbot. Therefore, I am currently updating the code and description of the current simulation robot and the actual robot to this repository together.

Each ROS package for simulation and real robot is placed separately in two folder.

1. ROS package for simulation robot: https://github.com/kimbring2/DeepSoccer/tree/master/deepsoccer_pc
2. ROS package for real robot: https://github.com/kimbring2/DeepSoccer/tree/master/deepsoccer_jetson
3. Simulation to Real to connect 1, 2 method: https://github.com/kimbring2/DeepSoccer/blob/master/sim2real

You can run each part separately. However, two part should be connected by sim2real method.

# 4. Reference
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

# 5. Citation
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

# 6. Acknowledgement
<img src="image/POM_Jetson.png"> <strong>I receive a prize from NVIDIA for this project</strong>

<img src="image/Jetson_AI_Specialist.png"> <strong>I receive Jetson AI Specialist certification from NVIDIA by this project</strong>

# 7. License
Apache License 2.0
