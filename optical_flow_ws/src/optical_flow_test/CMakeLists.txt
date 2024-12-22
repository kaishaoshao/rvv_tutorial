
cmake_minimum_required(VERSION 3.13)

project(optical_flow_test)

# set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" ${CMAKE_MODULE_PATH})

# set(CMAKE_C_COMPILER "gcc") # 设置C编译器
# set(CMAKE_CXX_COMPILER "g++") # 设置C++编译器

# set(CMAKE_C_COMPILER "clang")
# set(CMAKE_CXX_COMPILER "clang++")

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if( NOT CMAKE_BUILD_TYPE )
  message(STATUS "1 CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}") 
  set( CMAKE_BUILD_TYPE Release)
else()
  message(STATUS "2 CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}") 
endif()
message(STATUS "3 CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# for gdb debug
# SET(CMAKE_BUILD_TYPE "Debug")
# # SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
# SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# SET(CMAKE_BUILD_TYPE "RelWithDebInfo")
# # set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g") # cmake default: "-O2 -g -DNDEBUG"
# set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

SET(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=rv64gcv")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=rv64gcv_zfh") # support vfloat16
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=rv64gcv_zfh_zvfh1p0") # support vfloat16
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=rv64gcv_zba_zbb_zbc_zbs -fno-tree-vectorize -fno-tree-loop-vectorize")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

set(Eigen3_DIR "/root/SETUP/eigen-3.4.0_install/share/eigen3/cmake")
set(Ceres_DIR "/root/SETUP/ceres210_install/lib/cmake/Ceres")
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
# find_package(glog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(TBB REQUIRED)

find_package(catkin REQUIRED COMPONENTS 
  roscpp
  rosbag
  cv_bridge
  image_transport
  nav_msgs
  sensor_msgs
  std_msgs
  roslib
  message_filters
  message_generation

  # sensor_pub
  # myodom
  # atp_info
  # gnss_comm

)

catkin_package(
  #INCLUDE_DIRS include
#  LIBRARIES binocular_sensor
  CATKIN_DEPENDS roscpp std_msgs rosmsg
#  DEPENDS system_lib
)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" ${CMAKE_MODULE_PATH}) # move here.

# Make specialization for double / float optional.
# Disabling them can be useful for better compile times during development.
option(BASALT_INSTANTIATIONS_DOUBLE "Instatiate templates for Scalar=double." ON)
option(BASALT_INSTANTIATIONS_FLOAT "Instatiate templates for Scalar=float." ON) 
if(BASALT_INSTANTIATIONS_DOUBLE)
  list(APPEND BASALT_COMPILE_DEFINITIONS BASALT_INSTANTIATIONS_DOUBLE)
endif()
if(BASALT_INSTANTIATIONS_FLOAT)
  list(APPEND BASALT_COMPILE_DEFINITIONS BASALT_INSTANTIATIONS_FLOAT)
endif()

set(EIGEN_INCLUDE_DIR_HINTS ${EIGEN_ROOT})
find_package(Eigen3 3.4.0 EXACT REQUIRED MODULE)
message(STATUS "Found Eigen headers in: ${EIGEN3_INCLUDE_DIR}")
if(NOT EIGEN3_INCLUDE_DIR MATCHES "^${EIGEN_ROOT}")
  message(WARNING "Found Eigen headers are outside of specified EIGEN_ROOT '${EIGEN_ROOT}'")
endif()

find_package(TBB REQUIRED)
message(STATUS "Found TBB ${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR} (interface version ${TBB_INTERFACE_VERSION}) headers in: ${TBB_INCLUDE_DIRS}")
if (TBB_INTERFACE_VERSION LESS 11004)
  # enable global_control header for earlier TBB versions (Ubuntu 16.04, 18.04)
  add_definitions(-DTBB_PREVIEW_GLOBAL_CONTROL)
endif()

# NOTE: not specifying version, since 2, 3 or 4 is fine
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d highgui)
message(STATUS "Found OpenCV ${OpenCV_VERSION} headers in: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Found OpenCV_LIBS: ${OpenCV_LIBS}")

# Add our own custom scoped opencv target since none is provided by OpenCV itself
add_library(basalt::opencv INTERFACE IMPORTED)
set_property(TARGET basalt::opencv PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OpenCV_INCLUDE_DIRS})
set_property(TARGET basalt::opencv PROPERTY INTERFACE_LINK_LIBRARIES ${OpenCV_LIBS})

find_package(fmt REQUIRED)
message(STATUS "Found {fmt} ${fmt_VERSION} in: ${fmt_DIR}")

add_subdirectory(thirdparty)

# custom scoped magic_enum target
add_library(basalt::magic_enum INTERFACE IMPORTED)
set_property(TARGET basalt::magic_enum PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/magic_enum/include)

include_directories(
  ${catkin_INCLUDE_DIRS}
#   /root/SETUP/gperftools_install/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# link_directories(/root/SETUP/gperftools_install/lib)
# link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_executable(optical_flow_test
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
  src/wx_ros1_io.cpp
  src/wx_yaml_io.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/keypoints.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/vio_config.cpp
  src/util/compile_date_time.cpp
  src/undistorted_photometric.cpp
  src/util/cmap.cpp
)


# target_link_libraries(test1 Eigen3::Eigen)

# target_link_libraries(test1 ${OpenCV_LIBS})

# target_link_libraries(test1 -lpthread)

target_link_libraries(optical_flow_test
  ${catkin_LIBRARIES}
  ${STD_CXX_FS} basalt::opencv basalt::basalt-headers TBB::tbb
  # basalt::magic_enum rosbag apriltag opengv nlohmann::json fmt::fmt 
  basalt::magic_enum opengv nlohmann::json fmt::fmt 
  yaml-cpp
  # Ceres::ceres
  ${CERES_LIBRARIES}
  # -lprofiler # for gperftools
  # -ltcmalloc
  glog::glog
  # loop_ros
  )

# add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_generate_messages_cpp) # for msg.

target_include_directories(optical_flow_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(optical_flow_test PUBLIC ${BASALT_COMPILE_DEFINITIONS})


