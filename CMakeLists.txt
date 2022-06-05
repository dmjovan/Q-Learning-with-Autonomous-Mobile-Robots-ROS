cmake_minimum_required(VERSION 3.0.2)
project(Q-Learning-with-Autonomous-Mobile-Robots-ROS)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)


catkin_python_setup()

generate_messages(
   DEPENDENCIES
   std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs 
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS 
  src/robotic_systems/training_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

