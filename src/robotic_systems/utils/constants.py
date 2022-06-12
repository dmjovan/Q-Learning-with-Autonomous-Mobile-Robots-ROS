# initial position if not random position
X_INIT = -0.4
Y_INIT = -0.4
THETA_INIT = 45.0

# Q-learning action/state constants
STATE_SPACE_IND_MAX = 143
STATE_SPACE_IND_MIN = 0

ANGLE_MAX = 359
ANGLE_MIN = 0
HORIZON_WIDTH = 75

ACTION_MAP = {0: "forward", 1: "left", 2: "right"}

# Q-learning speed parameters
CONST_LINEAR_SPEED_FORWARD = 0.08
CONST_ANGULAR_SPEED_FORWARD = 0.0
CONST_LINEAR_SPEED_TURN = 0.06
CONST_ANGULAR_SPEED_TURN = 0.4

# Feedback control parameters
K_RO = 2
K_ALPHA = 15
K_BETA = -3
V_CONST = 0.1  # [m/s]

PATH_IND = 4
INIT_POSITIONS_X = [-0.7, -0.7, -0.5, -1, -2]
INIT_POSITIONS_Y = [-0.7, 0.7, 1, -2, 1]
INIT_POSITIONS_THETA = [45, -45, -120, -90, 150]
GOAL_POSITIONS_X = [2.0, 2.0, 0.5, 1, 2]
GOAL_POSITIONS_Y = [1.0, -1.0, -1.9, 2, -1, ]
GOAL_POSITIONS_THETA = [25.0, -40.0, -40, 60, -30, ]

# Goal reaching threshold
GOAL_DIST_THRESHOLD = 0.1  # [m]
GOAL_ANGLE_THRESHOLD = 15  # [degrees]

# Lidar constants
MAX_LIDAR_DISTANCE = 1.0
COLLISION_DISTANCE = 0.14  # LaserScan.range_min = 0.1199999
NEARBY_DISTANCE = 0.45
ZONE_0_LENGTH = 0.4
ZONE_1_LENGTH = 0.7
ANGLE_MAX = 359
ANGLE_MIN = 0
HORIZON_WIDTH = 75

# Experiment versioning
EXPERIMENT_NAME = "experiment_01"

# Experiment paths
PROJECT_ROOT = "/home/ros/ROS_Workspace/ROS_Projects/src/Q-Learning-with-Autonomous-Mobile-Robots-ROS/src/robotic_systems"
EXPERIMENTS_DIR = "experiments"

EXPERIMENT_ROOT = f"{PROJECT_ROOT}/{EXPERIMENTS_DIR}/{EXPERIMENT_NAME}"
Q_TABLE_PATH = f"{EXPERIMENT_ROOT}/Q_table.csv"
STEPS_PER_EPISODE_PATH = f"{EXPERIMENT_ROOT}/steps_per_episode.csv"
REWARD_PER_EPISODE_PATH = f"{EXPERIMENT_ROOT}/reward_per_episode.csv"
EPSILON_PER_EPISODE_PATH = f"{EXPERIMENT_ROOT}/epsilon_per_episode.csv"
REWARD_MIN_PER_EPISODE_PATH = f"{EXPERIMENT_ROOT}/reward_min_per_episode.csv"
REWARD_MAX_PER_EPISODE_PATH = f"{EXPERIMENT_ROOT}/reward_max_per_episode.csv"
REWARD_AVG_PER_EPISODE_PATH = f"{EXPERIMENT_ROOT}/reward_avg_per_episode.csv"
