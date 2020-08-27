# Gazebo Solenoid Electromagnet Spring plugin 
This Gazebo/ROS plugin allows you to easily add a solenoid electromagnet spring to your robot in Gazebo.

### Installation
Place this foler to your catkin workspace and use 'catkin_make' command.

### Adding plugin
This plugin attaches to a joint, so the plugin needs to be given a reference to that link.

```
<robot>
  <joint name="stick" type="prismatic">
    <origin xyz="-0.281228 -0.278693 -0.136057" rpy="-5.12956e-06 7.06783e-08 -1.534" />
    <parent link="body_1" />
    <child link="roller_stick_1" />
    <axis xyz="-1 0 0"/>
    <limit lower="-0.02" upper="0.01" effort="30" velocity="30" />
    <joint_properties friction="1.0"/>
  </joint>
    
  <gazebo>
    <plugin name="stick_solenoid_electromagnet_joint_spring" filename="libsolenoid_electromagnet_joint_spring_plugin.so">
      <kx>1000</kx>
      <set_point>0.01</set_point>
      <joint>stick</joint>
    </plugin>
  </gazebo>
</robot>
```

### Parameters

``kx`` : The spring coefficient in N-m

``set_point`` : The pooint at which the solenoid electromagnet joint would feel no force 

``joint`` : Name of the joint to add the solenoid electromagnet to
