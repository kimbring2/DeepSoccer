# cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
# project(gazebo_joint_torsional_spring)


# find_package(gazebo REQUIRED)

# include_directories(${GAZEBO_INCLUDE_DIRS})
# link_directories(${catkin_INCLUDE_DIRS} ${GAZEBO_LIBRARY_DIRS})

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

# add_library(gazebo_joint_torsional_spring_plugin SHARED src/gazebo_joint_torsional_spring.cpp)
# target_link_libraries(gazebo_joint_torsional_spring_plugin ${GAZEBO_libraries})


cmake_minimum_required(VERSION 2.8.3)
project(gazebo_joint_torsional_spring)

# Load catkin and all dependencies required for this package
find_package(catkin REQUIRED COMPONENTS 
  roscpp 
  gazebo_ros 
)

# Depend on system install of Gazebo
find_package(gazebo REQUIRED)

link_directories(${GAZEBO_LIBRARY_DIRS})
include_directories(${Boost_INCLUDE_DIR} ${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})

list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

add_library(${PROJECT_NAME} src/gazebo_joint_torsional_spring.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

catkin_package(
  DEPENDS 
    roscpp 
    gazebo_ros 
)