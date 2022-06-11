#!/usr/bin/env python3

import os
import math
from platform import node
import rospy
import argparse
import numpy as np
from pathlib import Path

from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from robotic_systems.qlearning import QLearner
from robotic_systems.lidar_helper import LidarHelper

from robotic_systems.utils.constants import *
from robotic_systems.utils.rospy_utils import *


class InferenceNode:

    def __init__(self, random_pos: bool = False) -> None:

        rospy.init_node('InferenceNode', anonymous = False)
        self.rate = rospy.Rate(10)

        self.robot_position_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size = 10)
        self.action_pub = rospy.Publisher('/robot_action', String, queue_size = 10)

        self.random_init_pos = random_pos

        # Q-Learning algorithm instance
        self.qlearner = QLearner(True)

        # data & logs dir
        self.log_file_dir = os.path.join(PROJECT_ROOT, LOG_FILE_DIR)
        Path(self.log_file_dir).mkdir(exist_ok=True, parents=True)

        # subscriber to laser data
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        # initial robot positions & goal position
        self.x_init = INIT_POSITIONS_X[PATH_IND]
        self.y_init = INIT_POSITIONS_Y[PATH_IND]
        self.theta_init = INIT_POSITIONS_THETA[PATH_IND]

        self.x_goal = GOAL_POSITIONS_X[PATH_IND]
        self.y_goal = GOAL_POSITIONS_Y[PATH_IND]
        self.theta_goal = GOAL_POSITIONS_THETA[PATH_IND]

        # initial position
        self.robot_spawned = False

        # goal reached flag
        self.goal_reached = False


    def run(self):

        while not rospy.is_shutdown():
            self.rate.sleep()


    def check_initial_position(self, x_init: float, y_init: float, theta_init: float) -> bool:

        odomMsg = rospy.wait_for_message('/odom', Odometry)
        x, y = get_position(odomMsg)
        theta = math.degrees(get_rotation(odomMsg))

        if abs(x-x_init) < 0.01 and abs(y-y_init) < 0.01 and abs(theta-theta_init) < 1:
            return True
        else:
            return False


    def reset_position(self) -> tuple:

        """ 
            Set robot to initial position 
        """

        if self.random_init_pos:
            ckpt, x, y, theta = get_random_position()
        else:
            ckpt, x, y, theta = get_init_position(self.x_init, self.y_init, self.theta_init)

        self.robot_position_pub.publish(ckpt)

        return x, y, theta


    def scan_callback(self, lidarMsg: LaserScan):
        
        if self.goal_reached:
            self.action_pub.publish(String("stop"))
            rospy.signal_shutdown('Goal reached! End of testing!')

        # spawning robot on initial position
        if not self.robot_spawned:

            rospy.loginfo("Spawning robot...")

            # stopping the robot
            self.action_pub.publish(String("stop"))

            x_init, y_init, theta_init = self.reset_position()
            self.robot_spawned = self.check_initial_position(x_init, y_init, theta_init)

            rospy.loginfo(f"Robot spawned on initial position: x = {x_init}, y = {y_init}, theta = {theta_init}")
            rospy.loginfo(f"Robots goal position is: x = {self.x_goal}, y = {self.y_goal}, theta = {self.theta_goal}")

        else:
            odomMsg = rospy.wait_for_message('/odom', Odometry)

            # get robot position and orientation
            x, y = get_position(odomMsg)
            theta = math.degrees(get_rotation(odomMsg))

            # get lidar scan
            # taking measurements and deriving them
            lidar, angles = LidarHelper.scan_to_arr(lidarMsg)
            state_ind, x1, x2, x3, x4 = LidarHelper.discretize_lidar_scan(self.qlearner.state_space, lidar)
            
            # check crash or object nearby or goal nearby
            crash = LidarHelper.check_crash(lidar)
            object_nearby = LidarHelper.check_object_nearby(lidar)
            goal_near = LidarHelper.check_goal_near(x, y, self.x_goal, self.y_goal)

            # stop the testing
            if crash:
                self.action_pub.publish(String("stop"))
                rospy.loginfo("Crash! Stopping robot. Terminating inference node.")
                rospy.signal_shutdown('Crash! End of testing!')

            # feedback control algorithm -> if there is a clear path to the goal
            elif not object_nearby or goal_near:
                rospy.loginfo("Feedback robot control ...")
                v_scal, w_scal, self.goal_reached = feedback_control(x, y, theta, self.x_goal, self.y_goal, math.radians(self.theta_goal))
                self.action_pub.publish(String(str(v_scal) + "_" + str(w_scal)))

            # Q-learning algorithm inference -> 
            else:
                rospy.loginfo("Q-Learning inference for robot control ...")
                action = self.qlearner.get_best_action(state_ind)
                self.action_pub.publish(ACTION_MAP[action])


if __name__ == '__main__':

    try:
        arg_formatter = argparse.ArgumentDefaultsHelpFormatter
        parser = argparse.ArgumentParser(formatter_class=arg_formatter)
        parser.add_argument("--random-pos", dest="random_pos", type=bool, default=False)

        args = parser.parse_args(rospy.myargv()[1:])

        random_pos = True if str(args.random_pos) in ["True", "true"] else False

        node = InferenceNode(random_pos=random_pos)
        node.run()

    except rospy.ROSInterruptException:
        node.action_pub.publish(String("stop"))
        rospy.loginfo("Inference node terminated!")
        pass
