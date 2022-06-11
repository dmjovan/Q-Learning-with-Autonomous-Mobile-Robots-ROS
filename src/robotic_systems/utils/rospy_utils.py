import rospy
import math
import numpy as np

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from robotic_systems.utils.constants import *

    
def get_rotation(odomMsg: Odometry) -> float:

    """
        Get yaw angle in radians from provided odomMsg.
    """

    orientation_q = odomMsg.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    roll, pitch, yaw = euler_from_quaternion(orientation_list)
    return yaw


def get_position(odomMsg: Odometry) -> tuple:

    """
        Get x and y coordinates in meters.
    """

    x = odomMsg.pose.pose.position.x
    y = odomMsg.pose.pose.position.y
    return x, y

    
def get_lin_vel(odomMsg: Odometry) -> float:

    """
        Get linear speed (on x-axis) in m/s.
    """

    return odomMsg.twist.twist.linear.x


    
def get_ang_vel(odomMsg: Odometry) -> float:

    """
        Get angular speed in rad/s over z-axis.
    """

    return odomMsg.twist.twist.angular.z


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

        
def check_stability(k_rho: float, k_alpha: float, k_beta: float) -> bool:

    """
        Checking stability condition for FB control.
    """

    return k_rho > 0 and k_beta < 0 and k_alpha > k_rho


def check_strong_stability(k_rho: float, k_alpha: float, k_beta: float) -> bool:

    """
        Checking strong stability condition for FB control.
    """

    return k_rho > 0 and k_beta < 0 and k_alpha + 5 * k_beta / 3 - 2 * k_rho / np.pi > 0

def get_random_position() -> tuple:

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

    return checkpoint, x, y, theta

def get_init_position() -> tuple:

    """
        Setting robot position.
    """

    checkpoint = ModelState()

    checkpoint.model_name = 'turtlebot3_burger'

    checkpoint.pose.position.x = X_INIT
    checkpoint.pose.position.y = Y_INIT
    checkpoint.pose.position.z = 0.0

    [x_q, y_q, z_q, w_q] = quaternion_from_euler(0.0, 0.0, math.radians(THETA_INIT))

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

    return checkpoint, X_INIT, Y_INIT, THETA_INIT


def feedback_control(x, y, theta, x_goal, y_goal, theta_goal) -> tuple:
    """ Feedback (FB) robot control algorithm """

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
        goal_reached = True
        v = 0
        w = 0
        v_scal = 0
        w_scal = 0
    else:
        goal_reached = False
        v = K_RO * rho
        w = K_ALPHA * alpha + K_BETA * beta
        v_scal = v / abs(v) * V_CONST
        w_scal = w / abs(v) * V_CONST

    return v_scal, w_scal, goal_reached
    
