
cmake_minimum_required(VERSION 3.13)

project(test1)

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
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)


SET(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=rv64gcv")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=rv64gcv_zba_zbb_zbc_zbs -fno-tree-vectorize -fno-tree-loop-vectorize")


add_executable(test1
#   ${CMAKE_CURRENT_SOURCE_DIR}/src/test1.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/rvv_saxpy.cpp
)



