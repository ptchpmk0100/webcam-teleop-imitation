cmake_minimum_required(VERSION 3.20)
project(teleop_il_rollout CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MuJoCo's C library at the pinned release. By default the copy inside the Python wheel (mujoco==4.0.1), so Python
# and C++ run the very same library build; pass -DMUJOCO_DIR=<folder with include/ and libmujoco.so.*> to override.
if(NOT MUJOCO_DIR)
  message(FATAL_ERROR "set MUJOCO_DIR (the parity script passes the Python wheel's mujoco folder)")
endif()
file(GLOB MUJOCO_LIB "${MUJOCO_DIR}/libmujoco.so.*")
if(NOT MUJOCO_LIB)
  message(FATAL_ERROR "no libmujoco.so.* in ${MUJOCO_DIR}")
endif()

add_executable(rollout rollout.cc)
target_include_directories(rollout PRIVATE "${MUJOCO_DIR}/include")
target_link_libraries(rollout PRIVATE ${MUJOCO_LIB})
set_target_properties(rollout PROPERTIES BUILD_RPATH "${MUJOCO_DIR}")
target_compile_options(rollout PRIVATE -Wall -Wextra -O2)
