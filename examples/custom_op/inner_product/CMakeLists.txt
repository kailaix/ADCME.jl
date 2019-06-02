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
# include_directories(/home/kailai/anaconda3/lib/python3.6/site-packages/tensorflow/include)
# link_directories(/home/kailai/anaconda3/lib/python3.6/site-packages/tensorflow)
include_directories(/anaconda3/lib/python3.7/site-packages/tensorflow/include)
link_directories(/anaconda3/lib/python3.7/site-packages/tensorflow)

add_library(inner_product SHARED inner_product.cc)
add_library(inner_product_grad SHARED inner_product_grad.cc)
set_property(TARGET inner_product PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET inner_product_grad PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(inner_product tensorflow_framework)
target_link_libraries(inner_product_grad tensorflow_framework)
