#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

from .rospy_utils import *
from .constants import *


class RobotControllerNode:

    def __init__(self) -> None:

        rospy.init_node('RobotControllerNode', anonymous=False)
        self.rate = rospy.Rate(10)

        # publishers
        self.robot_velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # subscribers
        rospy.Subscriber('/robot_action', String, self.action_callback)

    def run(self) -> None:

        while not rospy.is_shutdown():
            self.rate.sleep()

    def action_callback(self, data) -> None:

        command = str(data.data)

        try:
            v_scal, w_scal = command.split("_")

            rospy.loginfo(f"Executing command V: {v_scal}, W: {w_scal}")

            vel_msg = create_vel_msg(float(v_scal), float(w_scal))
            self.robot_velocity_pub.publish(vel_msg)

        except ValueError:

            rospy.loginfo(f"Executing action {command.upper()}")
            getattr(self, f"do_{command}")()

    def do_forward(self) -> None:

        """
            Robot is going forward.
        """

        vel_msg = create_vel_msg(CONST_LINEAR_SPEED_FORWARD, CONST_ANGULAR_SPEED_FORWARD)
        self.robot_velocity_pub.publish(vel_msg)

    def do_left(self) -> None:

        """
            Robot is turning left.
        """

        vel_msg = create_vel_msg(CONST_LINEAR_SPEED_TURN, CONST_ANGULAR_SPEED_TURN)
        self.robot_velocity_pub.publish(vel_msg)

    def do_right(self) -> None:

        """
            Robot is turning right.
        """

        vel_msg = create_vel_msg(CONST_LINEAR_SPEED_TURN, -CONST_ANGULAR_SPEED_TURN)
        self.robot_velocity_pub.publish(vel_msg)

    def do_stop(self) -> None:

        """
            Robot is stopping.
        """

        vel_msg = create_vel_msg(0.0, 0.0)
        self.robot_velocity_pub.publish(vel_msg)

    def do_terminate(self) -> None:

        """
            Shutting down.
        """

        rospy.signal_shutdown("Terminating robot controller")