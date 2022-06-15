#!/usr/bin/env python3

import rospy
import argparse

from std_msgs.msg import String

from robotic_systems.utils.inference_node_utils import InferenceNode


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

        rospy.loginfo("Inference node terminated!")

