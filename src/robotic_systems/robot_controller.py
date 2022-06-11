#!/usr/bin/env python3

import math
import rospy
import numpy as np
from std_msgs.msg import String

from robotic_systems.utils.rospy_utils import *
from robotic_systems.utils.constants import *


class RobotControllerNode:

    def __init__(self) -> None:

        rospy.init_node('RobotControllerNode', anonymous = False)
        self.rate = rospy.Rate(10)
        
        # publishers
        self.robot_velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)

        #subscribers
        rospy.Subscriber('/robot_action', String, self.action_callback)


    def action_callback(self, data):

        rospy.loginfo(f"Executing action {str(data.data).upper()}")
        getattr(self, "do_", str(data.data))()


    def do_forward(self) -> None:

        """
            Robot is going forward.
        """

        vel_msg = create_vel_msg(CONST_LINEAR_SPEED_FORWARD, CONST_ANGULAR_SPEED_FORWARD)
        self.robot_velocity_pub.publish(vel_msg)


    def do_left(self):

        """
            Robot is turning left.
        """

        vel_msg = create_vel_msg(CONST_LINEAR_SPEED_TURN, CONST_ANGULAR_SPEED_TURN)
        self.robot_velocity_pub.publish(vel_msg)


    def do_right(self):

        """
            Robot is turning right.
        """

        vel_msg = create_vel_msg(CONST_LINEAR_SPEED_TURN, -CONST_ANGULAR_SPEED_TURN)
        self.robot_velocity_pub.publish(vel_msg)


    def do_stop(self):

        """
            Robot is stopping.
        """

        vel_msg = create_vel_msg(0.0, 0.0)
        self.robot_velocity_pub.publish(vel_msg)


    def feedback_control(self, x, y, theta, x_goal, y_goal, theta_goal) -> str:

        """
            Feedback (FB) robot control algorithm
        """

        # theta goal normalization
        if theta_goal >= np.pi:
            theta_goal_norm = theta_goal - 2 * np.pi
        else:
            theta_goal_norm = theta_goal

        rho = np.sqrt(pow((x_goal - x), 2) + pow((y_goal - y), 2))
        lamdba_ = math.atan2((y_goal - y), (x_goal - x ))

        alpha = (lamdba_ -  theta + np.pi) % (2 * np.pi) - np.pi
        beta = (theta_goal - lamdba_ + np.pi) % (2 * np.pi) - np.pi

        if rho < GOAL_DIST_THRESHOLD and math.degrees(abs(theta-theta_goal_norm)) < GOAL_ANGLE_THRESHOLD:
            status = 'Goal position reached!'
            v = 0
            w = 0
            v_scal = 0
            w_scal = 0
        else:
            status = 'Goal position not reached!'
            v = K_RO * rho
            w = K_ALPHA * alpha + K_BETA * beta
            v_scal = v / abs(v) * V_CONST
            w_scal = w / abs(v) * V_CONST

        vel_msg = create_vel_msg(v_scal, w_scal)
        self.robot_velocity_pub.publish(vel_msg)

        return status


if __name__ == "__main__":

    try:
        node = RobotControllerNode()
        node.run()

    except rospy.ROSInterruptException:
        rospy.loginfo("Robot controller node terminated!")
        pass

