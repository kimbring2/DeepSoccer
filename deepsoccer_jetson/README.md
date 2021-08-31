# 1. Software Dependency
- JetPack 4.4
- ROS Melodic
- Tensorflow 2.2.0
- Python2 for ROS acuator, sensor node
- Python3 for ROS main node

# 2. Prepare hardware
## Main Board
The main board used in DeepSoccer is Jetson Xavier NX which can run three Deep Learning model at the same time unlike Jetson Nano which can handle only one model one time. However, Jetson Xavier NX is little expensive compared to Jetson Nano. 

<img src="/image/Jetson-Xavier-NX-Developer-Kit-details-5.jpg" width="600">

This board can run ROS, Python, and Tensorflow. Therefore, you can develop yout own application easily. It is also very convenient to use sensors and actuators as it supports almost all communication interfaces such as I2C, SPI, Digital input / output, CSI and USB.

1. [NVIDIA Jetson Xavier NX](https://www.amazon.com/NVIDIA-Jetson-Xavier-Developer-812674024318/dp/B086874Q5R)

Purchases can generally be made through Amazon. Unlike the Jetson Nano, which can be operated with three 18650 batteries, the Jetson Xavier NX requires four 18650 batteries to operate stably. 

<img src="/image/NX_expansion.png" width="800">

2. [18650 Battery Holder 4 Slots](https://www.amazon.com/abcGoodefg-Battery-Holder-Plastic-Storage/dp/B071XTGBH6/ref=sr_1_1?crid=1L5HZHK1U0S6Y&dchild=1&keywords=18650+battery+holder+4+slot&qid=1606271098&sprefix=18650+battery+holder+4%2Caps%2C340&sr=8-1)
3. [Lithium Battery Protection Board](https://www.amazon.com/ZRM-Lithium-Protection-Overcharge-Electronic/dp/B07RGPSSQS/ref=sr_1_23?dchild=1&keywords=Lithium+Polymer+Battery+Protection+Board&qid=1606271336&sr=8-23)
4. [Lithium Battery Voltage Indicator](https://www.amazon.com/1S-3-7V-Battery-Voltage-Tester/dp/B01N9M05DA/ref=pd_lpo_328_img_1/130-4493602-6974653?_encoding=UTF8&pd_rd_i=B01N9M05DA&pd_rd_r=beb7e722-d92f-489e-810c-7578d1397195&pd_rd_w=vCrGR&pd_rd_wg=WfnfD&pf_rd_p=7b36d496-f366-4631-94d3-61b87b52511b&pf_rd_r=ZYPH4HRXPAE1124HNKJ3&psc=1&refRID=ZYPH4HRXPAE1124HNKJ3)
5. [L298 Motor Driver](https://www.amazon.com/HiLetgo-Controller-Stepper-H-Bridge-Mega2560/dp/B07BK1QL5T/ref=sr_1_3?crid=3T1MYM4DCAHQ0&dchild=1&keywords=l298+motor+driver&qid=1606271045&sprefix=l298+%2Caps%2C361&sr=8-3)

## OLED
<img src="/image/deepsoccer_oled_1.jpg" width="800">

DeepSoccer has OLED display like an original Jetbot to monitor IP Address, memory usage without monitor connection.

1. [Waveshare OLED module](https://www.waveshare.com/0.91inch-oled-module.htm)

Jetson Xavier NX is connected to OLED module by using VDC, GND and SCL, SDA of 0 Channel I2C.

<img src="/image/deepsoccer_oled_2.png" width="800">

After connecting the hardware, download the Jetbot package from https://github.com/NVIDIA-AI-IOT/jetbot to Jetson Xaiver NX and install it using setup.py file. In this package, execute a python file(https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/jetbot/utils/create_stats_service.py) that displays the current information in OLED.

After that, try to register sevice to execut OLED file automatically when the board boot. First, move to /etc/systemd/system/ location of Ubuntu. Then, create a file named deepsoccer_stats.service with following contents.

```
[Unit]
Description=DeepSoccer stats display service
[Service]
Type=simple
User=kimbring2
ExecStart=/bin/sh -c "python3 /home/kimbring2/jetbot/jetbot/apps/stats.py"
Restart=always
[Install]
WantedBy=multi-user.target
```

Then, register the file as a service and start it as shown below.

```
$ systemctl daemon-reload
$ systemctl enable deepsoccer_stats
$ systemctl start deepsoccer_stats
```

The registered service can be confirmed with the following command.

```
sudo systemctl status deepsoccer_stats
```

## Wheel
<img src="/image/NX_Dynamixel.png" width="800">

The U2D2 board and U2D2 power hub can be purchased at the Robotis shopping mall. However, if you have existing one, you can change only the 12V power supply method and use the rest as it is.

1. [U2D2](http://www.robotis.us/u2d2/)
2. [U2D2 Power Hub](http://www.robotis.us/u2d2-power-hub-board-set/)
3. [Dynamixel MX-12W](https://www.robotis.us/dynamixel-mx-12w/)
4. [Omniwheel(Korea local shop)](http://robomecha.co.kr/product/detail.html?product_no=10&cate_no=1&display_group=2)

ID and communication method and firmware version of of Dynamixel can given via a program provided by ROBOTIS. I test a motor seperately before installing it to robot body. A power supply and TTL communication can be done by using a U2D2 board and power hub.

## Roller
The mechanism for controlling the ball is composed of a rubber roller for fixing and a solenoid electromagnet for kicking.

<img src="/image/NX_DC_Motor.png" width="800">

1. [Engraving rubber roller(Made in Korea)](
https://www.ebay.com/itm/50mm-Engraving-Rubber-Roller-Brayer-Stamping-Printing-Screening-Tool-Korea-/153413802463)

For the part for grabiing the soccer ball, I use a part of engraving roller. The core of the roller is made by 3D printer and connected to a DC motor which is included in origin Jetbot.

## Solenoid
For the part for kicking the soccer ball, I use a solenoid electromagnet. To control the solenoid electromagnet in the ROS, we should control GPIO using code. I use a GPIO library of Jetson (https://github.com/NVIDIA/jetson-gpio) provided by NVIDIA.

It is determined that directly connecting the solenoid motor directly to the 12V power supply can not enough force to kick the ball far. Thus, large capacity capacitor and charging circuit for it is added. Thankfully I could find circuit and component for this at https://drive.google.com/file/d/17twkN9F0Dghrc06b_U9Iviq9Si9rsyOG/view.

<img src="/image/NX_Solenoid.png" width="800">

1. [Large Capacity Capacitor](https://www.aliexpress.com/item/32866139188.html?spm=a2g0s.9042311.0.0.2db94c4dNsaPDZ)
2. [Large Capacitor Charger](https://www.aliexpress.com/item/32904490215.html?spm=a2g0s.9042311.0.0.27424c4dANjLyy)
3. [Limit switch(Relay module can replace it)](https://www.aliexpress.com/item/32860423798.html?spm=a2g0s.9042311.0.0.2db94c4dNsaPDZ)

After a 250v 1000uf capacitor and a ±45V-390V capacitor charger are added, a solenoid can push a heavy a billiard ball to considerable distance.

## Seosor
In addition to the actuators, the robot must observe the current state to decide which action is best at now. For that, DeepSocce robot is equipped with an infrared sensor to check whether robot is holding a ball using roller front side obstacle to prevent crash.

<img src="/image/NX_Sensor.png" width="800">

1. [Infrared Sensor](https://ko.aliexpress.com/item/32391592655.html)
2. [Ridar Sensor](https://www.adafruit.com/product/3978)

The infrared sensor is needed to be connected to the GPIO for digital signals on the Jetson board because it gives 0 and 1 signals. The Lidar sensor which send a signal as UART format is connected to the GPIO of Jetson for UART.

# 3. Usage
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

# 4. Teleoperation test
Like the original version of Jetbot, Jetbot soccer version can be controlled by gamepad. You can check a code for that [teleoperation_soccer.ipynb](https://github.com/kimbring2/DeepSoccer/blob/master/etc/teleoperation_soccer.ipynb) file. Upload it to Jetson Nano and run it.

As with the original Jetbot, you can use a Jupyter Notebook to test a your code without connecting with monitor. First, insall Jupyter package and creates a configuration file using the following command.

```
$ pip3 install jupyterlab
$ jupyter notebook --generate-config
```

Next, open ipython and generate a hash to set the password.

```
$ ipython

In [1]: from IPython.lib import passwd

In [2]: passwd()
Enter password: 
Verify password: 
Out[2]: 'sha1:60f3ac9aec93:be2d6048e9b1e7ae0f1ccbad9d746734bf5c3797'
```

Next, record generated hash in the jupyter_notebook_config.json file created at previous step.

```
$ sudo vi ~/.jupyter/jupyter_notebook_config.json

c = get_config()
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8080
c.NotebookApp.password = 'sha1:60f3ac9aec93:be2d6048e9b1e7ae0f1ccbad9d746734bf5c3797'
```

Finally, start Jupyter Notebook with the command below and enter the password you set earlier.

```
$ jupyter notebook
```

[![DeepSoccer teleoperation test](https://img.youtube.com/vi/vONoIruznlw/hqdefault.jpg)](https://www.youtube.com/watch?v=vONoIruznlw "Jetbot Soccer Play - Click to Watch!")
<strong>Click to Watch!</strong>

You can use the gamepad for performing the basic actions for soccer. Multi players will be able to play robot soccer together if power of robot is a bit more reinforced. It is still not enough to play real soccer.

# 5. Using pretrained model at Jetson board 
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
