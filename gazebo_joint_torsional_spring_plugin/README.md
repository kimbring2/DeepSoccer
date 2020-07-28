# Gazebo Joint Torsional Spring Plugin 

This Gazebo/ROS plugin allows you to easily add a torsional spring to your robot in Gazebo.

### Installation

```shell
cd ~/catkin_ws/src
git clone git@github.com:aminsung/gazebo_joint_torsional_spring_plugin.git
cd ~/catkin_ws
catkin_make
```

### Adding the joint torsional spring plugin

This plugin attaches to a joint, so the plugin needs to be given a reference to that link.

```xml
<robot>
  <joint name="knee_joint">
    ... joint description ...
  </joint>
    
  <gazebo>
    <!-- joint torsional spring plugin -->
    <plugin name="knee_joint_torsional_spring" filename="libgazebo_joint_torsional_spring.so">
      <kx>0.1</kx>
      <set_point>0.5</set_point>
      <joint>knee_joint</joint>
    </plugin>
  </gazebo>
    
</robot>
```

### Parameters

``kx`` : The spring coefficient in N-m

``set_point`` : The angle at which the joint would feel no force in radians

``joint`` : Name of the joint to add the torsional spring to

### Contribute

If you find a bug in the code, feel free to submit a pull request.