#!/usr/bin/env python3

import rospy
import argparse

from std_msgs.msg import String

from utils.training_node_utils import TrainingNode
        
        
if __name__ == '__main__':

    # initializing node variable
    node = None

    try:
        arg_formatter = argparse.ArgumentDefaultsHelpFormatter
        parser = argparse.ArgumentParser(formatter_class=arg_formatter)
        parser.add_argument("--load-q-table", dest="load_q_table", type=bool, default=False)
        parser.add_argument("--max-episodes", dest="max_episodes", type=int, default=500)
        parser.add_argument("--max-steps", dest="max_steps", type=int, default=500)
        parser.add_argument("--random-pos", dest="random_pos", type=bool, default=False)
        parser.add_argument("--alpha", dest="alpha", type=float, default=0.5)
        parser.add_argument("--gamma", dest="gamma", type=float, default=0.9)
        parser.add_argument("--epsilon", dest="epsilon", type=float, default=0.9)
        parser.add_argument("--epsilon-grad", dest="epsilon_grad", type=float, default=0.96)
        parser.add_argument("--epsilon-min", dest="epsilon_min", type=float, default=0.05)

        args = parser.parse_args(rospy.myargv()[1:])

        load_q_table = True if str(args.load_q_table) in ["True", "true"] else False
        max_episodes = int(args.max_episodes)
        max_steps = int(args.max_steps)
        random_pos = True if str(args.random_pos) in ["True", "true"] else False
        alpha = float(args.alpha)
        gamma_ = float(args.gamma)
        epsilon = float(args.epsilon)
        epsilon_grad = float(args.epsilon_grad)
        epsilon_min = float(args.epsilon_min)

        node = TrainingNode(load_q_table=load_q_table,
                            max_episodes=max_episodes,
                            max_steps_per_episode=max_steps,
                            random_init_position_flag=random_pos,
                            alpha=alpha,
                            gamma=gamma_,
                            epsilon=epsilon,
                            epsilon_grad=epsilon_grad,
                            epsilon_min=epsilon_min)

        rospy.loginfo("Starting Training node!")
        node.run()

    except rospy.ROSInterruptException:

        rospy.loginfo("Training node terminated!")
        node.save_data()
