import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import deepsoccer_env_test
from gym.envs.registration import register
from geometry_msgs.msg import Point
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os

from cv_bridge import CvBridge, CvBridgeError
import cv2
bridge = CvBridge()


class DeepSoccerSingleTestEnv(deepsoccer_env_test.DeepSoccerTestEnv):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot2 in some kind of maze.
        It will learn how to move around the maze without crashing.
        """

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/deepsoccer/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="jetbot_gazebo",
                    launch_file_name="robots_soccer_test.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/deepsoccer/config",
                               yaml_file_name="deepsoccer_single_test.yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(DeepSoccerSingleTestEnv, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here
        number_actions = rospy.get_param('/deepsoccer_test/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

        #number_observations = rospy.get_param('/turtlebot2/n_observations')
        """
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """

        self.kick_flag = 0

        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/deepsoccer_test/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/deepsoccer_test/linear_turn_speed')
        self.angular_speed = rospy.get_param('/deepsoccer_test/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/deepsoccer_test/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/deepsoccer_test/init_linear_turn_speed')

        self.new_ranges = rospy.get_param('/deepsoccer_test/new_ranges')
        self.min_range = rospy.get_param('/deepsoccer_test/min_range')
        self.max_laser_value = rospy.get_param('/deepsoccer_test/max_laser_value')
        self.min_laser_value = rospy.get_param('/deepsoccer_test/min_laser_value')

        # Get Desired Point to Get
        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/deepsoccer_test/desired_pose/x")
        self.desired_point.y = rospy.get_param("/deepsoccer_test/desired_pose/y")
        self.desired_point.z = rospy.get_param("/deepsoccer_test/desired_pose/z")

        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        #laser_scan = self.get_laser_scan()
        #rospy.logdebug("laser_scan len===>" + str(len(laser_scan.ranges)))

        #num_laser_readings = int(len(laser_scan.ranges)/self.new_ranges)
        num_laser_readings = 1
        high = numpy.full((num_laser_readings), self.max_laser_value)
        low = numpy.full((num_laser_readings), self.min_laser_value)

        # We only use two integers
        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Rewards
        self.forwards_reward = rospy.get_param("/deepsoccer_test/forwards_reward")
        self.turn_reward = rospy.get_param("/deepsoccer_test/turn_reward")
        self.end_episode_points = rospy.get_param("/deepsoccer_test/end_episode_points")

        self.cumulated_steps = 0.0

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base(0, 0, 0, 0)

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0

        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

        model_state = self.get_model_state()
        robot1_index = (model_state.name).index("robot1")
        robot1_pose = (model_state.pose)[robot1_index]
        #odometry = self.get_odom()
        #self.previous_distance_from_des_point = self.get_distance_from_desired_point(odometry.pose.pose.position)
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(robot1_pose.position)

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.

        stop_action = [0, 0, 0, 0, 30, 80]
        forward_action = [-30, 30, 30, -30, 30, 0]
        left_action = [30, 30, 30, 30, 30, 0]
        right_action = [-30, -30, -30, -30, 30, 0]
        bacward_action = [30, -30, -30, 30, 30, 0]
        hold_action = [0, 0, 0, 0, 30, 0]
        kick_action = [0, 0, 0, 0, 0, -80]
        """

        rospy.logdebug("Start Set Action ==>" + str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0: #STOP
            self.kick_flag = 0
            self.move_base(0, 0, 0, 0)
            self.last_action = "STOP"
        elif action == 1: #FORWARD
            self.move_base(-50, 50, 50, -50)
            self.last_action = "FORWARD"
        elif action == 2: #LEFT
            self.move_base(50, 50, 50, 50)
            self.last_action = "LEFT"
        elif action == 3: #RIGHT
            self.move_base(-50, -50, -50, -50)
            self.last_action = "RIGHT"
        elif action == 4: #BACKWARD
            self.move_base(50, -50, -50, 50)
            self.last_action = "BACKWARD"
        elif action == 5: #HOLD
            self.move_base(0, 0, 0, 0)
            self.last_action = "HOLD"
        elif action == 6: #KICK
            if self.kick_flag == 0:
                self.move_base(0, 0, 0, 0)
                self.kick_flag = 1
            elif self.kick_flag == 1:
                self.move_base(0, 0, 0, 0)

            self.last_action = "KICK"

        # We tell TurtleBot2 the linear and angular speed to set to execute
        #self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)

        rospy.logdebug("END Set Action ==>" + str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        #laser_scan = self.get_laser_scan()

        #camera_frame = self.get_camera_rgb_image_raw()
        #try:
        #    cv_image = bridge.imgmsg_to_cv2(camera_frame, "passthrough")
        #except CvBridgeError, e:
        #    rospy.logerr("CvBridge Error: {0}".format(e))

        #new_cv_image = cv2.resize(cv_image, (512, 512), interpolation=cv2.INTER_AREA)
        #cv2.imshow("Image of Robot Camera", new_cv_image)
        #cv2.waitKey(3)

        #discretized_laser_scan = self.discretize_observation(self.new_ranges)

        # We get the odometry so that SumitXL knows where it is.
        #odometry = self.get_odom()
        #x_position = odometry.pose.pose.position.x
        #y_position = odometry.pose.pose.position.y
        model_state = self.get_model_state()
        robot1_index = (model_state.name).index("robot1")
        robot1_pose = (model_state.pose)[robot1_index]
        x_position = robot1_pose.position.x
        y_position = robot1_pose.position.y

        # We round to only two decimals to avoid very big Observation space
        #odometry_array = [round(x_position, 2),round(y_position, 2)]
        model_state_array = [round(x_position, 2),round(y_position, 2)]

        # We only want the X and Y position and the Yaw
        #observations = discretized_laser_scan + odometry_array
        #observations = discretized_laser_scan

        observations = [1]
        rospy.logdebug("Observations==>" + str(observations))
        rospy.logdebug("END Get Observation ==>")
        return observations

    def _is_done(self, observations):
        if self._episode_done:
            rospy.logerr("DeepSoccer is Too Close to wall==>")
        else:
            rospy.logerr("DeepSoccer didnt crash at least ==>")

            model_state = self.get_model_state()
            robot1_index = (model_state.name).index("robot1")
            football_index = (model_state.name).index("football")
            left_goal_index = (model_state.name).index("left_goal")
            right_goal_index = (model_state.name).index("right_goal")

            robot1_pose = (model_state.pose)[robot1_index]
            football_pose = (model_state.pose)[football_index]
            left_goal_pose = (model_state.pose)[left_goal_index]
            right_goal_pose = (model_state.pose)[right_goal_index]

            #print("football_pose.position.x: " + str(football_pose.position.x))
            #print("left_goal_pose.position.x: " + str(left_goal_pose.position.x))
            #print("left_goal_pose.position.y: " + str(left_goal_pose.position.y))
            if (football_pose.position.x < left_goal_pose.position.x):
                self._episode_done = True
            elif (football_pose.position.x > right_goal_pose.position.x):
                self._episode_done = True

        return self._episode_done

    def _compute_reward(self, observations, done):
        model_state = self.get_model_state()
        robot1_index = (model_state.name).index("robot1")
        football_index = (model_state.name).index("football")
        left_goal_index = (model_state.name).index("left_goal")
        right_goal_index = (model_state.name).index("right_goal")

        robot1_pose = (model_state.pose)[robot1_index]
        football_pose = (model_state.pose)[football_index]
        left_goal_pose = (model_state.pose)[left_goal_index]
        right_goal_pose = (model_state.pose)[right_goal_index]

        #print("football_pose.position.x: " + str(football_pose.position.x))
        #print("left_goal_pose.position.x: " + str(left_goal_pose.position.x))
        #print("left_goal_pose.position.y: " + str(left_goal_pose.position.y))
        reward = 0
        if (football_pose.position.x < left_goal_pose.position.x):
            reward = 1
        elif (football_pose.position.x > right_goal_pose.position.x):
            reward = -1

        '''
        if not done:
            if self.last_action == "FORWARDS":
                reward = self.forwards_reward
            else:
                reward = self.turn_reward

            # If there has been a decrease in the distance to the desired point, we reward it
            if distance_difference < 0.0:
                rospy.logwarn("DECREASE IN DISTANCE GOOD")
                reward += self.forwards_reward
            else:
                rospy.logerr("ENCREASE IN DISTANCE BAD")
                reward += 0
        else:
            if self.is_in_desired_position(current_position):
                reward = self.end_episode_points
            else:
                reward = -1 * self.end_episode_points

        self.previous_distance_from_des_point = distance_from_des_point
        '''
        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward

    # Internal TaskEnv Methods
    def discretize_observation(self, data, new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        discretized_ranges = []
        mod = len(data.ranges) / new_ranges

        #rospy.logwarn("data=" + str(data))
        rospy.logwarn("new_ranges=" + str(new_ranges))
        rospy.logwarn("mod=" + str(mod))
        '''
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or numpy.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif numpy.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    discretized_ranges.append(int(item))

                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" + str(item) + "< " + str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.logwarn("NOT done Validation >>> item=" + str(item) + "< " + str(self.min_range))
        '''
        #print("discretized_ranges: " + str(discretized_ranges))
        lidar_range = data.ranges[360]
        if lidar_range == float("inf"):
            lidar_range = 12

        discretized_ranges = [lidar_range]

        return discretized_ranges

    def is_in_desired_position(self, current_position, epsilon=0.05):
        """
        It return True if the current position is similar to the desired poistion
        """
        is_in_desired_pos = False

        x_pos_plus = self.desired_point.x + epsilon
        x_pos_minus = self.desired_point.x - epsilon
        y_pos_plus = self.desired_point.y + epsilon
        y_pos_minus = self.desired_point.y - epsilon

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close

        return is_in_desired_pos

    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.desired_point)

        return distance

    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        return distance

