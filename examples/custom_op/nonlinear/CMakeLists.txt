cmake_minimum_required(VERSION 3.5)

# set(CMAKE_CXX_COMPILER /usr/bin/gcc-5)

# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

# https://github.com/tensorflow/tensorflow/issues/1569
# if GCC > 5
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 ${CMAKE_CXX_FLAGS}")
endif()


# import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))
# import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))

# check the system python and pycall python are the same
# build the actual operation which can be used directory
include_directories(/anaconda3/lib/python3.7/site-packages/tensorflow/include)
link_directories(/anaconda3/lib/python3.7/site-packages/tensorflow)

add_library(nonlinear SHARED nonlinear.cpp)
set_property(TARGET nonlinear PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(nonlinear tensorflow_framework)
