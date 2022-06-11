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
V_CONST = 0.1 # [m/s]

# Goal reaching threshold
GOAL_DIST_THRESHOLD = 0.1 # [m]
GOAL_ANGLE_THRESHOLD = 15 # [degrees]

PROJECT_ROOT = "/home/ros/ROS_Workspace/ROS_Projects/src/Q-Learning-with-Autonomous-Mobile-Robots-ROS/src/robotic_systems"
LOG_FILE_DIR = "data"
Q_TABLE_PATH = f"{PROJECT_ROOT}/{LOG_FILE_DIR}/Q_table.csv"
STEPS_PER_EPISODE_PATH = f"{PROJECT_ROOT}/{LOG_FILE_DIR}/steps_per_episode.csv"
REWARD_PER_EPISODE_PATH = f"{PROJECT_ROOT}/{LOG_FILE_DIR}/reward_per_episode.csv"
EPSILON_PER_EPISODE_PATH = f"{PROJECT_ROOT}/{LOG_FILE_DIR}/epsilon_per_episode.csv"
REWARD_MIN_PER_EPISODE_PATH = f"{PROJECT_ROOT}/{LOG_FILE_DIR}/reward_min_per_episode.csv"
REWARD_MAX_PER_EPISODE_PATH = f"{PROJECT_ROOT}/{LOG_FILE_DIR}/reward_max_per_episode.csv"
REWARD_AVG_PER_EPISODE_PATH = f"{PROJECT_ROOT}/{LOG_FILE_DIR}/reward_avg_per_episode.csv"


