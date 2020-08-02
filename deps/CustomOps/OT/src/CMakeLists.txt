cmake_minimum_required(VERSION 3.5)
project(TFApp)

set (CMAKE_CXX_STANDARD 11)

execute_process(COMMAND julia -e "import ADCME; print(ADCME.__STR__)" OUTPUT_VARIABLE JL_OUT)
list(GET JL_OUT 0 BINDIR)
list(GET JL_OUT 1 LIBDIR)
list(GET JL_OUT 2 TF_INC)
list(GET JL_OUT 3 TF_ABI)
list(GET JL_OUT 4 PREFIXDIR)
list(GET JL_OUT 5 CC)
list(GET JL_OUT 6 CXX)
list(GET JL_OUT 7 CMAKE)
list(GET JL_OUT 8 MAKE)
list(GET JL_OUT 9 GIT)
list(GET JL_OUT 10 PYTHON)
list(GET JL_OUT 11 TF_LIB_FILE)

message("Python path=${PYTHON}")
message("PREFIXDIR=${PREFIXDIR}")
message("TF_INC=${TF_INC}")
message("TF_ABI=${TF_ABI}")
message("TF_LIB_FILE=${TF_LIB_FILE}")

# # C++11 required for tensorflow
# set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
# execute_process(COMMAND julia -e "using PyCall; print(PyCall.python)" OUTPUT_VARIABLE PYTHON)
# message("Python path=${PYTHON}")
# execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_compile_flags()[0][2:])" OUTPUT_VARIABLE TF_INC)
# execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_link_flags()[0][2:])" OUTPUT_VARIABLE TF_LIB)
# execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_compile_flags()[1][-1])" OUTPUT_VARIABLE TF_ABI)
# execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_link_flags()[1][3:])" OUTPUT_VARIABLE TF_LIB_FILE)


# # https://github.com/tensorflow/tensorflow/issues/1569
# # if GCC > 5
# if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
#   set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${TF_ABI} ${CMAKE_CXX_FLAGS}")
# endif()

message("${PREFIXDIR}")
include_directories(${TF_INC} ${PREFIXDIR})

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O3 -DNDEBUG")


# add_library(SparseIndexing SHARED SparseIndexing.cpp)
# set_property(TARGET SparseIndexing PROPERTY POSITION_INDEPENDENT_CODE ON)
# target_link_libraries(SparseIndexing ${TF_LIB_FILE} "${TORCH_LIBRARIES}")

add_executable(main test.cpp sinkhorn.cpp)
set_property(TARGET main PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(main ${TF_LIB_FILE})
