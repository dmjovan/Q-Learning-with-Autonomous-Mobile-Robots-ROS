#!/usr/bin/env python3

import rospy
import random 
import numpy as np
from itertools import product

from robotic_systems.utils.constants import *

class QLearner:

    def __init__(self, load_q_table: bool = False, 
                       alpha: float = 0.5, 
                       gamma: float = 0.9, 
                       epsilon: float = 0.9,
                       epsilon_grad: float = 0.96,
                       epsilon_min: float = 0.05) -> None:
        
        self.actions = self.create_action_space()
        self.state_space = self.create_state_space()

        if not load_q_table:
            self.Q_table = self.create_Q_table()
            rospy.loginfo("Created Q table")
        else:
            try:
                self.Q_table = self.read_Q_table(path=Q_TABLE_PATH)
                rospy.loginfo("Loaded Q table")

            except FileNotFoundError:
                self.Q_table = self.create_Q_table()
                rospy.loginfo("Created Q table")
        

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_grad = epsilon_grad
        self.epsilon_min = epsilon_min


    def create_action_space(self) -> np.ndarray:
        return np.array([0,1,2]) # 0 -> straight forward, 1 -> turn left, 2 -> turn right


    def create_state_space(self) -> np.ndarray:
        x1 = set((0,1,2))
        x2 = set((0,1,2))
        x3 = set((0,1,2,3))
        x4 = set((0,1,2,3))
        state_space = set(product(x1,x2,x3,x4))
        return np.array(list(state_space))


    def create_Q_table(self) -> np.ndarray:
        return np.zeros((self.state_space.size, self.actions.size))


    def read_Q_table(self, path: str) -> np.ndarray:
        return np.genfromtxt(path, delimiter = ' , ')


    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_grad)


    def get_random_action(self) -> int:
        return random.choice(self.actions)


    def get_best_action(self, state_ind: int) -> int:

        if STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX:
            action = self.actions[np.argmax(self.Q_table[state_ind,:])]
        else:
            action = self.get_random_action()
        return action


    def epsilon_greedy_exploration(self, state_ind: int):
        if random.random() > self.epsilon and STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX:
            action= self.get_best_action(state_ind)
        else:
            action = self.get_random_action()

        return action


    def get_reward(self, action: int, 
                         prev_action: int, 
                         lidar: np.ndarray, 
                         prev_lidar: np.ndarray, crash: bool) -> float:

        if crash:
            reward = -100

        else:

            left_half_indices_start = ANGLE_MIN + HORIZON_WIDTH
            left_half_indices_end = ANGLE_MIN

            right_half_indices_start = ANGLE_MAX
            right_half_indices_end = ANGLE_MAX - HORIZON_WIDTH

            lidar_horizon = np.concatenate((lidar[left_half_indices_start:left_half_indices_end:-1],
                                            lidar[right_half_indices_start:right_half_indices_end:-1]))

            prev_lidar_horizon = np.concatenate((prev_lidar[left_half_indices_start:left_half_indices_end:-1],
                                                 prev_lidar[right_half_indices_start:right_half_indices_end:-1]))

            # reward from taken action
            if action == 0:
                r_action = 0.2 # forward -> 0.2 reward
            else:
                r_action = -0.1 # turning -> -0.1 rewards

            # reward from crash distance to obstacle change
            W = np.linspace(0.9, 1.1, len(lidar_horizon) // 2)
            W = np.append(W, np.linspace(1.1, 0.9, len(lidar_horizon) // 2))

            if np.sum(W * ( lidar_horizon - prev_lidar_horizon)) >= 0:
                r_obstacle = 0.2
            else:
                r_obstacle = -0.2

            # reward from turn left/right change
            if (prev_action == 1 and action == 2) or (prev_action == 2 and action == 1):
                r_change = -0.8 # getting bigger punishment for moving left-right-left-right
            else:
                r_change = 0.0

            # overall step reward
            reward = r_action + r_obstacle + r_change

        return reward


    def update_Q_table(self, state_ind: int, 
                             action: int, 
                             reward: float, 
                             next_state_ind: int) -> None:

        if STATE_SPACE_IND_MIN <= state_ind <= STATE_SPACE_IND_MAX and STATE_SPACE_IND_MIN <= next_state_ind <= STATE_SPACE_IND_MAX:
            self.Q_table[state_ind, action] = (1 - self.alpha) * self.Q_table[state_ind, action] + \
                                              self.alpha * ( reward + self.gamma * max(self.Q_table[next_state_ind,:]) )
        else:
            raise("Invalid state index")
