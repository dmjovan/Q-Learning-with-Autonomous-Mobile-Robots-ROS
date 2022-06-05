#!/usr/bin/env python3

import math
import numpy as np
from sensor_msgs.msg import LaserScan

MAX_LIDAR_DISTANCE = 1.0
COLLISION_DISTANCE = 0.14 # LaserScan.range_min = 0.1199999
NEARBY_DISTANCE = 0.45

ZONE_0_LENGTH = 0.4
ZONE_1_LENGTH = 0.7

ANGLE_MAX = 359
ANGLE_MIN = 0
HORIZON_WIDTH = 75

class LidarHelper:

    def __init__(self) -> None:
        pass

    @staticmethod
    def scan_to_arr(msgScan: LaserScan) -> tuple(np.ndarray, np.ndarray):

        """
            Convert LasecScan msg to array of distances and angles.
        """

        distances = []
        angles = []

        for i in range(len(msgScan.ranges)):
            angle = math.degrees(i * msgScan.angle_increment)

            if msgScan.ranges[i] > MAX_LIDAR_DISTANCE:
                distance = MAX_LIDAR_DISTANCE

            elif msgScan.ranges[i] < msgScan.range_min:

                distance = msgScan.range_min

                # for real robot protection
                if msgScan.ranges[i] < 0.01:
                    distance = MAX_LIDAR_DISTANCE
            else:
                distance = msgScan.ranges[i]

            distances.append(distance)
            angles.append(angle)

        # distances in [m], angles in [degrees]
        return np.array(distances), np.array(angles)


    @staticmethod
    def discretize_lidar_scan(state_space: np.ndarray, lidar: np.ndarray) -> tuple(int, int, int, int, int):

        """
            Discretization of lidar measurements.
        """

        x1 = 2 # Left zone (no obstacle detected)
        x2 = 2 # Right zone (no obstacle detected)
        x3 = 3 # Left sector (no obstacle detected)
        x4 = 3 # Right sector (no obstacle detected)

        # Find the left side lidar values of the vehicle
        lidar_left = min(lidar[(ANGLE_MIN):(ANGLE_MIN + HORIZON_WIDTH)])
        if ZONE_1_LENGTH > lidar_left > ZONE_0_LENGTH:
            x1 = 1 # zone 1
        elif lidar_left <= ZONE_0_LENGTH:
            x1 = 0 # zone 0

        # Find the right side lidar values of the vehicle
        lidar_right = min(lidar[(ANGLE_MAX - HORIZON_WIDTH):(ANGLE_MAX)])
        if ZONE_1_LENGTH > lidar_right > ZONE_0_LENGTH:
            x2 = 1 # zone 1
        elif lidar_right <= ZONE_0_LENGTH:
            x2 = 0 # zone 0

        # Detection of object in front of the robot
        if ( min(lidar[(ANGLE_MAX - HORIZON_WIDTH // 3):(ANGLE_MAX)]) < 1.0 ) or ( min(lidar[(ANGLE_MIN):(ANGLE_MIN + HORIZON_WIDTH // 3)]) < 1.0 ):
            object_front = True
        else:
            object_front = False

        # Detection of object on the left side of the robot
        if min(lidar[(ANGLE_MIN):(ANGLE_MIN + 2 * HORIZON_WIDTH // 3)]) < 1.0:
            object_left = True
        else:
            object_left = False

        # Detection of object on the right side of the robot
        if min(lidar[(ANGLE_MAX - 2 * HORIZON_WIDTH // 3):(ANGLE_MAX)]) < 1.0:
            object_right = True
        else:
            object_right = False

        # Detection of object on the far left side of the robot
        if min(lidar[(ANGLE_MIN + HORIZON_WIDTH // 3):(ANGLE_MIN + HORIZON_WIDTH)]) < 1.0:
            object_far_left = True
        else:
            object_far_left = False

        # Detection of object on the far right side of the robot
        if min(lidar[(ANGLE_MAX - HORIZON_WIDTH):(ANGLE_MAX - HORIZON_WIDTH // 3)]) < 1.0:
            object_far_right = True
        else:
            object_far_right = False

        # The left sector of the vehicle
        if ( object_front and object_left ) and ( not object_far_left ):
            x3 = 0 # sector 0
        elif ( object_left and object_far_left ) and ( not object_front ):
            x3 = 1 # sector 1
        elif object_front and object_left and object_far_left:
            x3 = 2 # sector 2

        if ( object_front and object_right ) and ( not object_far_right ):
            x4 = 0 # sector 0
        elif ( object_right and object_far_right ) and ( not object_front ):
            x4 = 1 # sector 1
        elif object_front and object_right and object_far_right:
            x4 = 2 # sector 2

        # Find the state space index of (x1, x2, x3, x4) in Q table
        ss = np.where(np.all(state_space == np.array([x1, x2, x3, x4]), axis = 1))
        state_ind = int(ss[0])

        return state_ind, x1, x2, x3, x4


    @staticmethod
    def check_crash(lidar: np.ndarray) -> bool:

        """
            Checking if lidar detected crash.
        """

        lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
        W = np.linspace(1.2, 1, len(lidar_horizon) // 2)
        W = np.append(W, np.linspace(1, 1.2, len(lidar_horizon) // 2))
        if np.min( W * lidar_horizon ) < COLLISION_DISTANCE:
            return True
        else:
            return False


    @staticmethod
    def check_object_nearby(lidar: np.ndarray) -> bool:

        """
            Checking if lidar detected object nearby.
        """

        lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
        W = np.linspace(1.4, 1, len(lidar_horizon) // 2)
        W = np.append(W, np.linspace(1, 1.4, len(lidar_horizon) // 2))
        if np.min( W * lidar_horizon ) < NEARBY_DISTANCE:
            return True
        else:
            return False
