from typing import Optional
import math
import numpy as np

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from robotic_systems.utils.constants import *


def get_rotation_from_gps(modelStateMsg: ModelState) -> float:

    """
        Get yaw angle in radians from provided odomMsg.
    """

    orientation_q = modelStateMsg.pose[2].orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    roll, pitch, yaw = euler_from_quaternion(orientation_list)
    return yaw


def get_position_from_gps(modelStateMsg: ModelState) -> tuple:

    """
        Get x and y coordinates in meters.
    """

    x = modelStateMsg.pose[2].position.x
    y = modelStateMsg.pose[2].position.y
    return x, y

    
def get_rotation_from_odom(odomMsg: Odometry) -> float:

    """
        Get yaw angle in radians from provided odomMsg.
    """

    orientation_q = odomMsg.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    roll, pitch, yaw = euler_from_quaternion(orientation_list)
    return yaw


def get_position_from_odom(odomMsg: Odometry) -> tuple:

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

def get_init_position(x_init: Optional[float] = None,
                      y_init: Optional[float] = None,
                      theta_init: Optional[float] = None) -> tuple:

    """
        Setting robot position.
    """

    checkpoint = ModelState()

    checkpoint.model_name = 'turtlebot3_burger'

    x_init = x_init if x_init is not None else X_INIT
    y_init = y_init if y_init is not None else Y_INIT
    theta_init = theta_init if theta_init is not None else THETA_INIT

    checkpoint.pose.position.x = x_init
    checkpoint.pose.position.y = y_init
    checkpoint.pose.position.z = 0.0

    [x_q, y_q, z_q, w_q] = quaternion_from_euler(0.0, 0.0, math.radians(theta_init))

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

    return checkpoint, x_init, y_init, theta_init


def feedback_control(x, y, theta, x_goal, y_goal, theta_goal) -> tuple:
    """ Feedback (FB) robot control algorithm """

    # theta goal normalization
    if theta_goal >= np.pi:
        theta_goal_norm = theta_goal - 2 * np.pi
    else:
        theta_goal_norm = theta_goal

    rho = np.sqrt((x_goal - x)**2 + (y_goal - y)**2)
    lamdba_ = math.atan2((y_goal - y), (x_goal - x ))

    alpha = (lamdba_ -  theta + np.pi) % (2 * np.pi) - np.pi
    beta = (theta_goal - lamdba_ + np.pi) % (2 * np.pi) - np.pi

    goal_reached = False

    if rho < GOAL_DIST_THRESHOLD:
        goal_reached = True
        v = 0
        w = 0
        v_scal = 0
        w_scal = 0
    else:
        v = K_RO * rho
        w = K_ALPHA * alpha #+ K_BETA * beta
        v_scal = v / abs(v) * V_CONST
        w_scal = w / abs(v) * V_CONST

        if v > 0:
            v_scal = np.clip(v, V_CONST, 5*V_CONST)
        else:
            v_scal = np.clip(v, -5*V_CONST, -V_CONST)
        if w > 0:
            w_scal = np.clip(w, 0.5*V_CONST, 20*V_CONST)
        else:
            w_scal = np.clip(w, -20*V_CONST, -0.5*V_CONST)



    print(f"Distance from goal: {rho} ->  Goal reached: {goal_reached}")

    return v_scal, w_scal, goal_reached
    
