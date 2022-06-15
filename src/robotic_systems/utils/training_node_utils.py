#!/usr/bin/env python3

import os
import rospy
import numpy as np
from pathlib import Path

from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from robotic_systems.utils.qlearning import QLearner
from robotic_systems.utils.lidar_utils import LidarHelper
from robotic_systems.utils.constants import *
from robotic_systems.utils.robot_utils import *


class TrainingNode:

    def __init__(self, load_q_table: bool = False,
                 max_episodes: int = 500,
                 max_steps_per_episode: int = 500,
                 min_time_between_actions: float = 0.0,
                 random_init_position_flag: bool = False,
                 alpha: float = 0.5,
                 gamma: float = 0.9,
                 epsilon: float = 0.9,
                 epsilon_grad: float = 0.96,
                 epsilon_min: float = 0.05) -> None:

        rospy.init_node('TrainingNode', anonymous=False)
        self.rate = rospy.Rate(10)

        self.robot_position_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        self.action_pub = rospy.Publisher('/robot_action', String, queue_size=10)

        # .experiments/experiment_<experiment_num> for saving configuration and results
        self.log_file_dir = EXPERIMENT_ROOT
        Path(self.log_file_dir).mkdir(exist_ok=True, parents=True)

        # subscriber to laser experiment
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)

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
        self.qlearner = QLearner(load_q_table, self.alpha, self.gamma, self.epsilon, self.epsilon_grad,
                                 self.epsilon_min)

        # experiment trackers
        self.ep_reward_lst = []  # episodic reward
        self.steps_per_episode = []  # steps used for episode, over episodes
        self.reward_per_episode = []  # overall reward over episodes
        self.reward_max_per_episode = []  # max. reward over episodes
        self.reward_min_per_episode = []  # min. reward over episodes
        self.reward_avg_per_episode = []  # avg. reward over episodes
        self.epsilon_per_episode = []  # epsilon parameter over episodes

        self.ep_steps = 0
        self.ep_reward = 0
        self.episode = 1
        self.crash = False

        # initial position
        self.robot_spawned = False
        self.first_action_taken = False

        # previous values
        self.prev_lidar = None
        self.prev_action = None
        self.prev_state_ind = None

    def run(self):

        # Save initial hyper-parameters
        self.save_hyperparameters()

        # train
        while not rospy.is_shutdown():
            self.rate.sleep()

    @staticmethod
    def check_initial_position(x_init: float, y_init: float, theta_init: float) -> bool:

        odomMsg = rospy.wait_for_message('/odom', Odometry)
        x, y = get_position(odomMsg)
        theta = math.degrees(get_rotation(odomMsg))

        if abs(x - x_init) < 0.01 and abs(y - y_init) < 0.01 and abs(theta - theta_init) < 1:
            return True
        else:
            return False

    def reset_position(self) -> tuple:

        """
            Set robot to initial position
        """

        if self.random_init_position_flag:
            ckpt, x, y, theta = get_random_position()
        else:
            ckpt, x, y, theta = get_init_position()

        self.robot_position_pub.publish(ckpt)

        return x, y, theta

    def save_hyperparameters(self) -> None:
        """ Save hyper-parameters of the experiment """

        content = ""
        content += f"max_episodes: {self.max_episodes}\n"
        content += f"max_steps_per_episode: {self.max_steps_per_episode}\n"
        content += f"min_time_between_actions: {self.min_time_between_actions}\n"
        content += f"random_init_position_flag: {self.random_init_position_flag}\n"
        content += f"alpha: {self.alpha}\n"
        content += f"gamma: {self.gamma}\n"
        content += f"epsilon: {self.epsilon}\n"
        content += f"epsilon_grad: {self.epsilon_grad}\n"
        content += f"epsilon_min: {self.epsilon_min}\n"

        # Save hyper-parameters
        with open(os.path.join(EXPERIMENT_ROOT, 'hyper-parameters.txt'), 'w') as f:
            f.write(content)

    def save_data(self) -> None:

        if self.episode % 50 or self.episode >= self.max_episodes:
            np.savetxt(Q_TABLE_PATH, self.qlearner.Q_table, delimiter=' , ')
            np.savetxt(STEPS_PER_EPISODE_PATH, self.steps_per_episode, delimiter=' , ')
            np.savetxt(REWARD_PER_EPISODE_PATH, self.reward_per_episode, delimiter=' , ')
            np.savetxt(EPSILON_PER_EPISODE_PATH, self.epsilon_per_episode, delimiter=' , ')
            np.savetxt(REWARD_MIN_PER_EPISODE_PATH, self.reward_min_per_episode, delimiter=' , ')
            np.savetxt(REWARD_MAX_PER_EPISODE_PATH, self.reward_max_per_episode, delimiter=' , ')
            np.savetxt(REWARD_AVG_PER_EPISODE_PATH, self.reward_avg_per_episode, delimiter=' , ')

    def scan_callback(self, lidarMsg: LaserScan) -> None:

        # end of learning -> maximum episodes reached
        if self.episode > self.max_episodes:

            self.action_pub.publish(String("terminate"))
            rospy.signal_shutdown("End of learning process - Shutting down Training node")

        else:

            # end of episode -> there has been either a crash or number of steps in this episode has reached the maximum
            if self.crash or self.ep_steps >= self.max_steps_per_episode:

                if self.crash:
                    rospy.loginfo("Crash happened!")

                else:
                    rospy.loginfo("Maximum steps per episode reached!")

                # stopping the robot
                self.action_pub.publish(String("stop"))

                # adding logs into lists
                self.steps_per_episode.append(self.ep_steps)
                self.reward_per_episode.append(self.ep_reward)
                self.reward_min_per_episode.append(np.min(self.ep_reward_lst))
                self.reward_max_per_episode.append(np.max(self.ep_reward_lst))
                self.reward_avg_per_episode.append(np.mean(self.ep_reward_lst))
                self.epsilon_per_episode.append(self.qlearner.epsilon)

                # reset variables
                self.ep_reward_lst = []
                self.ep_steps = 0
                self.ep_reward = 0
                self.crash = False
                self.robot_spawned = False
                self.first_action_taken = False
                self.qlearner.update_epsilon()

                rospy.loginfo(f"Finished episode {self.episode}/{self.max_episodes}")
                self.episode = self.episode + 1

            else:
                self.ep_steps = self.ep_steps + 1

                # spawning robot on initial position
                if not self.robot_spawned:

                    rospy.loginfo("Spawning robot...")

                    # stopping the robot
                    self.action_pub.publish(String("stop"))

                    self.ep_steps = self.ep_steps - 1
                    self.first_action_taken = False

                    x_init, y_init, theta_init = self.reset_position()

                    self.robot_spawned = self.check_initial_position(x_init, y_init, theta_init)

                    rospy.loginfo(
                        f"Robot spawned on initial position: x = {x_init}, y = {y_init}, theta = {theta_init}")

                # taking first action
                elif not self.first_action_taken:

                    rospy.loginfo("Taking first action")

                    # taking measurements and deriving them
                    lidar, angles = LidarHelper.scan_to_arr(lidarMsg)
                    state_ind, x1, x2, x3, x4 = LidarHelper.discretize_lidar_scan(self.qlearner.state_space, lidar)
                    self.crash = LidarHelper.check_crash(lidar)

                    # get next action using epsilon-greedy policy
                    action = self.qlearner.epsilon_greedy_exploration(state_ind)

                    # execute the selected action
                    self.action_pub.publish(String(ACTION_MAP[action]))

                    self.prev_lidar = lidar
                    self.prev_action = action
                    self.prev_state_ind = state_ind

                    self.first_action_taken = True

                # normal flow of the algorithm
                else:

                    rospy.loginfo(f"Executing step {self.ep_steps}/{self.max_steps_per_episode}")

                    # taking measurements and deriving them
                    lidar, angles = LidarHelper.scan_to_arr(lidarMsg)
                    state_ind, x1, x2, x3, x4 = LidarHelper.discretize_lidar_scan(self.qlearner.state_space, lidar)
                    self.crash = LidarHelper.check_crash(lidar)

                    # get next action using epsilon-greedy policy
                    action = self.qlearner.epsilon_greedy_exploration(state_ind)

                    # get reward
                    reward = self.qlearner.get_reward(action, self.prev_action, lidar, self.prev_lidar, self.crash)

                    # update Q-table
                    self.qlearner.update_q_table(self.prev_state_ind, action, reward, state_ind)

                    # get next action using epsilon-greedy policy
                    action = self.qlearner.epsilon_greedy_exploration(state_ind)

                    # execute the selected action
                    self.action_pub.publish(String(ACTION_MAP[action]))

                    self.ep_reward += reward
                    self.ep_reward_lst.append(reward)
                    self.prev_lidar = lidar
                    self.prev_action = action
                    self.prev_state_ind = state_ind

        self.save_data()
