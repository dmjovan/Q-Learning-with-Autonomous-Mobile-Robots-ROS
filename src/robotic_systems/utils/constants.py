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
CONST_ANGULAR_SPEED_TURN = 0.5

# Feedback control parameters
K_RO = .1
K_ALPHA = .3
K_BETA = -.1
V_CONST = 0.1  # [m/s]

PATH_IND = 0
INIT_POSITIONS_X = [-2, -2, 0.5]
INIT_POSITIONS_Y = [0, 0, -0.5]
INIT_POSITIONS_THETA = [-30, 30, 180]
GOAL_POSITIONS_X = [-0.5, -0.5, -0.5]
GOAL_POSITIONS_Y = [-0.5, 0.5, -0.5]
GOAL_POSITIONS_THETA = [0, 30, 180]

# Goal reaching threshold
GOAL_DIST_THRESHOLD = 0.2  # [m]
GOAL_ANGLE_THRESHOLD = 360  # [degrees]

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
EXPERIMENT_NAME = "experiment_08"

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
