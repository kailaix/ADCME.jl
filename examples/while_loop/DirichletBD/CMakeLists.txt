cmake_minimum_required(VERSION 3.5)
project(TFApp)

# For PyTorch
# execute_process(COMMAND ${JULIA} -e "joinpath(Base.find_package(\"ADCME\"), \"deps/Libraries/libtorch\")|>print" OUTPUT_VARIABLE PREFIX)
# execute_process(COMMAND ${JULIA} -e "joinpath(Base.find_package(\"ADCME\"), \"examples/custom_op/headers\")|>print" OUTPUT_VARIABLE HEADERS)
# execute_process(COMMAND ${JULIA} -e "joinpath(Base.find_package(\"ADCME\"), \"deps/Libraries\")|>print" OUTPUT_VARIABLE PREFIXDIR)
# execute_process(COMMAND ${JULIA} -e "joinpath(Base.find_package(\"ADCME\"), \"deps/Libraries/libtorch/include\")|>print" OUTPUT_VARIABLE TORCH_INC)
# set(CMAKE_PREFIX_PATH ${PREFIX})
# find_package(Torch REQUIRED)

set (CMAKE_CXX_STANDARD 11)
# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

execute_process(COMMAND python -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_compile_flags()[0][2:])" OUTPUT_VARIABLE TF_INC)
execute_process(COMMAND python -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_link_flags()[0][2:])" OUTPUT_VARIABLE TF_LIB)
execute_process(COMMAND python -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_compile_flags()[1][-1])" OUTPUT_VARIABLE TF_ABI)


# https://github.com/tensorflow/tensorflow/issues/1569
# if GCC > 5
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${TF_ABI} ${CMAKE_CXX_FLAGS}")
endif()

include_directories(${TF_INC} ${PREFIXDIR}/../../include ${PREFIXDIR} ${TORCH_INC})
link_directories(${TF_LIB})

add_library(DirichletBD SHARED DirichletBD.cpp)
set_property(TARGET DirichletBD PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(DirichletBD tensorflow_framework "${TORCH_LIBRARIES}")
