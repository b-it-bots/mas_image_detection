cmake_minimum_required(VERSION 2.8.3)
project(mas_image_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  mas_perception_libs
)

catkin_python_setup()

catkin_package()

install(DIRECTORY ros/launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/ros/launch
)
