cmake_minimum_required(VERSION 3.5)
project(TFApp)

execute_process(COMMAND julia -e "using Conda; print(joinpath(Conda.LIBDIR, \"Libraries\"))" OUTPUT_VARIABLE EIGEN_INC)

set (CMAKE_CXX_STANDARD 11)
# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
execute_process(COMMAND julia -e "using PyCall; print(PyCall.python)" OUTPUT_VARIABLE PYTHON)
message("Python path=${PYTHON}")
execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_compile_flags()[0][2:])" OUTPUT_VARIABLE TF_INC)
execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_link_flags()[0][2:])" OUTPUT_VARIABLE TF_LIB)
execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_compile_flags()[1][-1])" OUTPUT_VARIABLE TF_ABI)
execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_link_flags()[1][3:])" OUTPUT_VARIABLE TF_LIB_FILE)


# https://github.com/tensorflow/tensorflow/issues/1569
# if GCC > 5
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${TF_ABI} ${CMAKE_CXX_FLAGS}")
endif()

include_directories(${TF_INC} ${HEADERS} ${EIGEN_INC} ${TORCH_INC})
link_directories(${TF_LIB})


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O3 -DNDEBUG")


add_library(SparseMatMul SHARED SparseMatMul.cpp)
set_property(TARGET SparseMatMul PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(SparseMatMul ${TF_LIB_FILE} "${TORCH_LIBRARIES}")
