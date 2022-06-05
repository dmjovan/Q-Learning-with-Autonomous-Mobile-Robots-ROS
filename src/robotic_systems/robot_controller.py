#!/usr/bin/env python3

import math
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# Q-learning speed parameters
CONST_LINEAR_SPEED_FORWARD = 0.08
CONST_ANGULAR_SPEED_FORWARD = 0.0
CONST_LINEAR_SPEED_TURN = 0.06
CONST_ANGULAR_SPEED_TURN = 0.4

# Feedback control parameters
K_RO = 2
K_ALPHA = 15
K_BETA = -3
V_CONST = 0.1 # [m/s]

# Goal reaching threshold
GOAL_DIST_THRESHOLD = 0.1 # [m]
GOAL_ANGLE_THRESHOLD = 15 # [degrees]

class RobotController:

    def __init__(self) -> None:
        
        # publishers
        self.robot_position_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size = 10)
        self.robot_velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)

        
    @staticmethod
    def get_rotation(odomMsg: Odometry) -> float:

        """
            Get yaw angle in radians from provided odomMsg.
        """

        orientation_q = odomMsg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return yaw


    @staticmethod
    def get_position(odomMsg: Odometry) -> tuple:

        """
            Get x and y coordinates in meters.
        """

        x = odomMsg.pose.pose.position.x
        y = odomMsg.pose.pose.position.y
        return x, y


    @staticmethod
    def check_goal_near(x: float, y: float, x_goal: float, y_goal: float) -> bool:

        """
            Checking if current robot position is near by goal. 
        """

        rho = math.sqrt(pow(( x_goal - x), 2) + pow((y_goal - y), 2))
        if rho < 0.3:
            return True
        else:
            return False


    @staticmethod
    def get_lin_vel(odomMsg: Odometry) -> float:

        """
            Get linear speed (on x-axis) in m/s.
        """

        return odomMsg.twist.twist.linear.x


    @staticmethod
    def get_ang_vel(odomMsg: Odometry) -> float:

        """
            Get angular speed in rad/s over z-axis.
        """

        return odomMsg.twist.twist.angular.z

    @staticmethod
    def create_vel_msg(vx: float, wz: float) -> Twist:

        """
            Creating ROS Twist message for /cmd_vel topic.
        """

        vel_msg = Twist()
        vel_msg.linear.x = vx
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = wz
        return vel_msg


    def execute_action(self, action: int) -> None:

        """
            Performing an action.
        """

        if action == 0:
            self.go_forward()

        elif action == 1:
            self.turn_left()

        elif action == 2:
            self.turn_right()

        else:
            raise("Invalid action selected.")


    def go_forward(self) -> None:

        """
            Robot is going forward.
        """

        vel_msg = self.create_vel_msg(CONST_LINEAR_SPEED_FORWARD, CONST_ANGULAR_SPEED_FORWARD)
        self.robot_velocity_pub.publish(vel_msg)


    def turn_left(self):

        """
            Robot is turning left.
        """

        vel_msg = self.create_vel_msg(CONST_LINEAR_SPEED_TURN, CONST_ANGULAR_SPEED_TURN)
        self.robot_velocity_pub.publish(vel_msg)


    def turn_right(self):

        """
            Robot is turning right.
        """

        vel_msg = self.create_vel_msg(CONST_LINEAR_SPEED_TURN, -CONST_ANGULAR_SPEED_TURN)
        self.robot_velocity_pub.publish(vel_msg)


    def stop(self):

        """
            Robot is stopping.
        """

        vel_msg = self.create_vel_msg(0.0, 0.0)
        self.robot_velocity_pub.publish(vel_msg)


    def set_position(self, x: float, y: float, theta: float) -> tuple:

        """
            Setting robot position.
        """

        checkpoint = ModelState()

        checkpoint.model_name = 'turtlebot3_burger'

        checkpoint.pose.position.x = x
        checkpoint.pose.position.y = y
        checkpoint.pose.position.z = 0.0

        [x_q, y_q, z_q, w_q] = quaternion_from_euler(0.0, 0.0, math.radians(theta))

        checkpoint.pose.orientation.x = x_q
        checkpoint.pose.orientation.y = y_q
        checkpoint.pose.orientation.z = z_q
        checkpoint.pose.orientation.w = w_q

        checkpoint.twist.linear.x = 0.0
        checkpoint.twist.linear.y = 0.0
        checkpoint.twist.linear.z = 0.0

        checkpoint.twist.angular.x = 0.0
        checkpoint.twist.angular.y = 0.0
        checkpoint.twist.angular.z = 0.0

        self.robot_position_pub.publish(checkpoint)

        return x, y, theta


    def set_random_position(self) -> tuple:

        """
            Setting robot in random position andd orientation.
            There are 10 possible positions, so that robot cannot hit in obstacle 
            while spawning.
        """

        x_range = np.array([-0.4, 0.6, 0.6, -1.4, -1.4, 2.0, 2.0, -2.5, 1.0, -1.0])
        y_range = np.array([-0.4, 0.6, -1.4, 0.6, -1.4, 1.0, -1.0, 0.0, 2.0, 2.0])
        theta_range = np.arange(0, 360, 15)

        ind = np.random.randint(0,len(x_range))
        ind_theta = np.random.randint(0,len(theta_range))

        x = x_range[ind]
        y = y_range[ind]
        theta = theta_range[ind_theta]

        checkpoint = ModelState()

        checkpoint.model_name = 'turtlebot3_burger'

        checkpoint.pose.position.x = x
        checkpoint.pose.position.y = y
        checkpoint.pose.position.z = 0.0

        [x_q, y_q, z_q, w_q] = quaternion_from_euler(0.0, 0.0, math.radians(theta))

        checkpoint.pose.orientation.x = x_q
        checkpoint.pose.orientation.y = y_q
        checkpoint.pose.orientation.z = z_q
        checkpoint.pose.orientation.w = w_q

        checkpoint.twist.linear.x = 0.0
        checkpoint.twist.linear.y = 0.0
        checkpoint.twist.linear.z = 0.0

        checkpoint.twist.angular.x = 0.0
        checkpoint.twist.angular.y = 0.0
        checkpoint.twist.angular.z = 0.0

        self.robot_position_pub.publish(checkpoint)

        return x, y, theta

        
    @staticmethod
    def check_stability(k_rho: float, k_alpha: float, k_beta: float) -> bool:

        """
            Checking stability condition for FB control.
        """

        return k_rho > 0 and k_beta < 0 and k_alpha > k_rho

    
    @staticmethod
    def check_strong_stability(k_rho: float, k_alpha: float, k_beta: float) -> bool:

        """
            Checking strong stability condition for FB control.
        """

        return k_rho > 0 and k_beta < 0 and k_alpha + 5 * k_beta / 3 - 2 * k_rho / np.pi > 0
        

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

        vel_msg = self.create_vel_msg(v_scal, w_scal)
        self.robot_velocity_pub.publish(vel_msg)

        return status
