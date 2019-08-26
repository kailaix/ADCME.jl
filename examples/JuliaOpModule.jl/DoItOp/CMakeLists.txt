cmake_minimum_required(VERSION 3.5)
project(TFApp)

##  For PyTorch
# execute_process(COMMAND julia -e "abspath(joinpath(Base.find_package(\"ADCME\"), \"../../deps/Libraries/libtorch\"))|>print" OUTPUT_VARIABLE PREFIX)
# execute_process(COMMAND julia -e "abspath(joinpath(Base.find_package(\"ADCME\"), \"../../examples/custom_op/headers\"))|>print" OUTPUT_VARIABLE HEADERS)
# execute_process(COMMAND julia -e "abspath(joinpath(Base.find_package(\"ADCME\"), \"../../deps/Libraries\"))|>print" OUTPUT_VARIABLE EIGEN_INC)
# execute_process(COMMAND julia -e "abspath(joinpath(Base.find_package(\"ADCME\"), \"../../deps/Libraries/libtorch/include\"))|>print" OUTPUT_VARIABLE TORCH_INC)
# set(CMAKE_PREFIX_PATH ${PREFIX})
# find_package(Torch REQUIRED)

set (CMAKE_CXX_STANDARD 11)
# C++11 required for tensorflow
execute_process(COMMAND python -c "from sysconfig import get_paths as gp; import sys; sys.stdout.write(gp()['include'])" OUTPUT_VARIABLE PYTHON_INC)
execute_process(COMMAND julia -e "using PyCall; print(PyCall.libpython)" OUTPUT_VARIABLE PYTHON_LIB)
execute_process(COMMAND julia -e "abspath(joinpath(Sys.BINDIR, \"../lib\"))|>print" OUTPUT_VARIABLE JULIA_LIB)
execute_process(COMMAND julia -e "abspath(joinpath(Sys.BINDIR, \"../include/julia\"))|>print" OUTPUT_VARIABLE JULIA_INC)
add_definitions( -DJULIA_ENABLE_THREADING=1 )


execute_process(COMMAND python -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_compile_flags()[0][2:])" OUTPUT_VARIABLE TF_INC)
execute_process(COMMAND python -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_link_flags()[0][2:])" OUTPUT_VARIABLE TF_LIB)
execute_process(COMMAND python -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_compile_flags()[1][-1])" OUTPUT_VARIABLE TF_ABI)


# https://github.com/tensorflow/tensorflow/issues/1569
# if GCC > 5
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${TF_ABI} ${CMAKE_CXX_FLAGS}")
endif()

include_directories(${TF_INC} ${HEADERS} ${EIGEN_INC} ${TORCH_INC})
include_directories(${JULIA_INC} ${PYTHON_INC})
link_directories(${TF_LIB} ${JULIA_LIB})

add_library(DoItOp SHARED DoItOp.cpp)
set_property(TARGET DoItOp PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(DoItOp tensorflow_framework julia ${PYTHON_LIB} "${TORCH_LIBRARIES}")
