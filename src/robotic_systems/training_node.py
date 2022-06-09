#!/usr/bin/env python3

import os
import math
import rospy
import argparse
import numpy as np

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from robotic_systems.qlearning import QLearner
from robotic_systems.robot_controller import RobotController
from robotic_systems.lidar_helper import LidarHelper

# initial position if not random position
X_INIT = -0.4
Y_INIT = -0.4
THETA_INIT = 45.0

class TrainingNode:

    def __init__(self, load_q_table: bool = False, 
                       max_episodes: int = 400,
                       max_steps_per_episode: int = 500,
                       min_time_between_actions: float = 0.0,
                       random_init_position_flag: bool = False,
                       alpha: float = 0.5,
                       gamma: float = 0.9,
                       epsilon: float = 0.9,
                       epsilon_grad: float = 0.96,
                       epsilon_min: float = 0.05, ) -> None:

        rospy.init_node('TrainingNode', anonymous = False)
        self.rate = rospy.Rate(10)

        self.robot_controller = RobotController()

        self.log_file_dir = "/home/ros/ROS_Workspace/ROS_Projects/src/Q-Learning-with-Autonomous-Mobile-Robots-ROS/src/robotic_systems/results"

        if not os.path.exists(self.log_file_dir): 
            os.makedirs(self.log_file_dir)

        # parameters
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.min_time_between_actions = min_time_between_actions
        self.random_init_position_flag = random_init_position_flag
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_grad = epsilon_grad
        self.epsilon_min = epsilon_min

        # Q-Learning algorithm instance
        self.qlearner = QLearner(load_q_table, self.alpha, self.gamma, self.epsilon, self.epsilon_grad, self.epsilon_min)

        # data trackers
        self.ep_reward_lst = [] # episodic reward
        self.steps_per_episode = [] # steps used for episode, over episodes
        self.reward_per_episode = [] # overall reward over episodes
        self.reward_max_per_episode = [] # max. reward over episodes
        self.reward_min_per_episode = [] # min. reward over episodes
        self.reward_avg_per_episode = [] # avg. reward over episodes
        self.epsilon_per_episode = [] # epislon parameter over episodes
        self.t_per_episode = [] # episode durations overs episodes

    
    def save_data(self, path: str, data: np.ndarray) -> None: 
        np.savetxt(path, data, delimiter = ' , ')


    def check_initial_position(self, x_init: float, y_init: float, theta_init: float) -> bool:

        odomMsg = rospy.wait_for_message('/odom', Odometry)
        x, y = self.robot_controller.get_position(odomMsg)
        theta = math.degrees(self.robot_controller.get_rotation(odomMsg))

        if abs(x-x_init) < 0.01 and abs(y-y_init) < 0.01 and abs(theta-theta_init) < 1:
            return True
        else:
            return False


    def run(self) -> None:
        
        ep_steps = 0
        ep_reward = 0
        episode = 1
        crash = False
        
        # initial position
        robot_spawned = False
        first_action_taken = False

        # init time
        t_0 = rospy.Time.now()
        t_start = rospy.Time.now()

        # init timer
        while not (t_start > t_0):
            t_start = rospy.Time.now()

        t_ep = t_start
        t_step = t_start

        while not rospy.is_shutdown():
            lidarMSg = rospy.wait_for_message('/scan', LaserScan)

            # secure the minimum time interval between 2 actions
            # step_time = (rospy.Time.now() - t_step).to_sec()
            # if step_time > self.min_time_between_actions:
            #     t_step = rospy.Time.now()

            # end of learning -> maximum episodes reached
            if episode > self.max_episodes:

                # save data to file
                self.save_data(os.path.join(self.log_file_dir, "/Q_table.csv"), self.qlearner.Q_table)
                self.save_data(os.path.join(self.log_file_dir, "/steps_per_episode.csv"), self.steps_per_episode)
                self.save_data(os.path.join(self.log_file_dir, "/reward_per_episode.csv"), self.reward_per_episode)
                self.save_data(os.path.join(self.log_file_dir, "/epsilon_per_episode.csv"), self.epsilon_per_episode)
                self.save_data(os.path.join(self.log_file_dir, "/reward_min_per_episode.csv"), self.reward_min_per_episode)
                self.save_data(os.path.join(self.log_file_dir, "/reward_max_per_episode.csv"), self.reward_max_per_episode)
                self.save_data(os.path.join(self.log_file_dir, "/reward_avg_per_episode.csv"), self.reward_avg_per_episode)
                self.save_data(os.path.join(self.log_file_dir, "/t_per_episode.csv"), self.t_per_episode)

                rospy.signal_shutdown("End of learning process - Shutting down TrainingNode")

            else:
                ep_time = (rospy.Time.now() - t_ep).to_sec()

                # end of episode -> there has been either a crash or number of steps in this episode has reached the maximum
                if crash or ep_steps >= self.max_steps_per_episode:

                    if crash:
                        rospy.loginfo("Crash happened!")

                    else:
                        rospy.loginfo("Maximum steps per episode reached!")

                    # stopping the robot
                    self.robot_controller.stop()

                    # adding data into lists
                    self.steps_per_episode.append(ep_steps)
                    self.reward_per_episode.append(ep_reward)
                    self.reward_min_per_episode.append( np.min(self.ep_reward_lst))
                    self.reward_max_per_episode.append(np.max(self.ep_reward_lst))
                    self.reward_avg_per_episode.append( np.mean(self.ep_reward_lst))
                    self.epsilon_per_episode.append(self.qlearner.epsilon)
                    self.t_per_episode.append(ep_time)

                    # reseting variables
                    t_ep = rospy.Time.now()
                    self.ep_reward_lst = []
                    ep_steps = 0
                    ep_reward = 0
                    crash = False
                    robot_spawned = False
                    first_action_taken = False
                    self.qlearner.update_epsilon()

                    rospy.loginfo(f"Finished episode {episode}/{self.max_episodes}")
                    episode = episode + 1

                
                else:
                    ep_steps = ep_steps + 1

                    # spawning robot on initial position
                    if not robot_spawned:

                        rospy.loginfo("Spawning robot...")

                        # stopping the robot
                        self.robot_controller.stop()

                        ep_steps = ep_steps - 1
                        first_action_taken = False

                        if self.random_init_position_flag:
                            x_init, y_init, theta_init = self.robot_controller.set_random_position()

                        else:
                            x_init, y_init, theta_init = self.robot_controller.set_position(X_INIT, Y_INIT, THETA_INIT)

                        robot_spawned = self.check_initial_position(x_init, y_init, theta_init)

                        rospy.loginfo(f"Robot spawned on initial position: x = {x_init}, y = {y_init}, theta = {theta_init}")

                    # taking first action
                    elif not first_action_taken:

                        rospy.loginfo("Taking first action")

                        # taking measurements and deriving them
                        lidar, angles = LidarHelper.scan_to_arr(lidarMSg)
                        state_ind, x1, x2, x3, x4 = LidarHelper.discretize_lidar_scan(self.qlearner.state_space, lidar)
                        crash = LidarHelper.check_crash(lidar)

                        # get next action using epsilon-greedy policy
                        action = self.qlearner.epsilon_greedy_exploration(state_ind)

                        # execute the selected action
                        self.robot_controller.execute_action(action)

                        prev_lidar = lidar
                        prev_action = action
                        prev_state_ind = state_ind

                        first_action_taken = True

                    # normal flow of the algorithm
                    else:                      

                        rospy.loginfo(f"Executing step {ep_steps}/{self.max_steps_per_episode}")

                        # taking measurements and deriving them
                        lidar, angles = LidarHelper.scan_to_arr(lidarMSg)
                        state_ind, x1, x2, x3, x4 = LidarHelper.discretize_lidar_scan(self.qlearner.state_space, lidar)
                        crash = LidarHelper.check_crash(lidar)

                        # get reward
                        reward = self.qlearner.get_reward(action, prev_action, lidar, prev_lidar, crash)
                        
                        # update Q-table
                        self.qlearner.update_Q_table(prev_state_ind, action, reward, state_ind)

                        # get next action using epsilon-greedy policy
                        action = self.qlearner.epsilon_greedy_exploration(state_ind)

                        # execute the selected action
                        self.robot_controller.execute_action(action)

                        ep_reward += reward
                        self.ep_reward_lst.append(reward)
                        prev_lidar = lidar
                        prev_action = action
                        prev_state_ind = state_ind

        

if __name__ == '__main__':

    try:
        arg_formatter = argparse.ArgumentDefaultsHelpFormatter
        parser = argparse.ArgumentParser(formatter_class=arg_formatter)
        parser.add_argument("--load-q-table", dest="load_q_table", type=bool, default=False)
        parser.add_argument("--max-episodes", dest="max_episodes", type=int, default=400)
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

        rospy.loginfo("Starting training node!")
        node.run()

    except rospy.ROSInterruptException:
        rospy.loginfo("Training node terminated!")
        pass
