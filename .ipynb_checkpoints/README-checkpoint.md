# Q-Learning-with-Autonomous-Mobile-Robots-ROS
 Implementation of Q-Learning algorithm for TurtleBot3 in ROS environment
 
 For running the training run the following commands (for the last command, check possible arguments in `training_node.py`):
 - export TURTLEBOT3_MODEL=burger 
 - roslaunch turtlebot3_gazebo turtlebot3_world.launch
 - rosrun Q-Learning-with-Autonomous-Mobile-Robots-ROS robot_controller_node.py
 - rosrun Q-Learning-with-Autonomous-Mobile-Robots-ROS training_node.py --max-episodes=2000 --max-steps=200 --random-pos=True
 
 For continuing some of the previous training run the following commands (check the last commnad):
 - export TURTLEBOT3_MODEL=burger 
 - roslaunch turtlebot3_gazebo turtlebot3_world.launch
 - rosrun Q-Learning-with-Autonomous-Mobile-Robots-ROS robot_controller_node.py
 - specify EXPERIMENT_ROOT in utils/constants.py where the desired Q table is stored
 - rosrun Q-Learning-with-Autonomous-Mobile-Robots-ROS training_node.py --load-q-table=True --max-episodes=2000 --max-steps=200 --random-pos=True
 
 For running the inference of trained model
 - export TURTLEBOT3_MODEL=burger 
 - roslaunch turtlebot3_gazebo turtlebot3_world.launch
 - rosrun Q-Learning-with-Autonomous-Mobile-Robots-ROS robot_controller_node.py
 - specify EXPERIMENT_ROOT in utils/constants.py where the desired Q table is stored
 - rosrun Q-Learning-with-Autonomous-Mobile-Robots-ROS inference_node.py --random-pos=True
 
