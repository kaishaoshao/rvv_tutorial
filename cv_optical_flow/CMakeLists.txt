
cmake_minimum_required(VERSION 3.13)

project(test1)

set(CMAKE_C_COMPILER "gcc") # 设置C编译器
set(CMAKE_CXX_COMPILER "g++") # 设置C++编译器

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

include_directories(
#   /root/SETUP/gperftools_install/include
${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# link_directories(/root/SETUP/gperftools_install/lib)

SET(CMAKE_BUILD_TYPE "Debug")
# # SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# SET(CMAKE_BUILD_TYPE "RelWithDebInfo")
# # set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g") # cmake default: "-O2 -g -DNDEBUG"
# set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

# SET(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=rv64gcv")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=rv64gcv_zfh") # support vfloat16
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=rv64gcv_zfh_zvfh1p0") # support vfloat16
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=rv64gcv_zba_zbb_zbc_zbs -fno-tree-vectorize -fno-tree-loop-vectorize")


# find_package(Eigen3 REQUIRED)
# find_package(OpenCV REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(TBB REQUIRED)

# test1 for sse
# add_executable(test1
#   ${CMAKE_CURRENT_SOURCE_DIR}/src/ReduceGray5x5.cpp
# )

# test2 for rvv
# add_executable(test2
#   ${CMAKE_CURRENT_SOURCE_DIR}/src/RVVReduceGray5x5.cpp
# )

add_executable(test1
${CMAKE_CURRENT_SOURCE_DIR}/src/test1.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/wx_lkpyramid.cpp
)

# add_executable(test3
#   ${CMAKE_CURRENT_SOURCE_DIR}/src/test3.cpp
# )

# target_link_libraries(test1 Eigen3::Eigen)

target_link_libraries(test1 ${OpenCV_LIBS} TBB::tbb)
# target_link_libraries(test2 ${OpenCV_LIBS})

# target_link_libraries(test1 -lpthread)


