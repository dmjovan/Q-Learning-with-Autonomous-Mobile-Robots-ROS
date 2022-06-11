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


    def run(self):

        while not rospy.is_shutdown():
            self.rate.sleep()


    def action_callback(self, data):

        command = str(data.data)

        if str(data.data) in ["forward", "left", "right", "stop"]:
            rospy.loginfo(f"Executing action {command.upper()}")
            getattr(self, f"do_{command}")()
            
        else:
            v_scal, w_scal = command.split("_")

            rospy.loginfo(f"Executing command V: {v_scal}, W: {w_scal}")

            vel_msg = create_vel_msg(float(v_scal), float(w_scal))
            self.robot_velocity_pub.publish(vel_msg)



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



if __name__ == "__main__":

    try:
        node = RobotControllerNode()
        node.run()

    except rospy.ROSInterruptException:
        rospy.loginfo("Robot controller node terminated!")
        pass

