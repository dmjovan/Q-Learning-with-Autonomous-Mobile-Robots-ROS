#!/usr/bin/env python3

import rospy

from robotic_systems.utils.robot_controller_utils import RobotControllerNode


if __name__ == "__main__":

    try:
        node = RobotControllerNode()
        node.run()

    except rospy.ROSInterruptException:
        rospy.loginfo("Robot controller node terminated!")
        pass

