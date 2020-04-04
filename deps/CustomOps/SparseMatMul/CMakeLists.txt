cmake_minimum_required(VERSION 3.5)
project(TFApp)

set (CMAKE_CXX_STANDARD 11)

execute_process(COMMAND julia -e "import ADCME; print(ADCME.__STR__)" OUTPUT_VARIABLE JL_OUT)
execute_process(COMMAND which julia OUTPUT_VARIABLE JULIA_BIN)
message("Julia=${JULIA_BIN}")

list(GET JL_OUT 0 BINDIR)
list(GET JL_OUT 1 LIBDIR)
list(GET JL_OUT 2 TF_INC)
list(GET JL_OUT 3 TF_ABI)
list(GET JL_OUT 4 EIGEN_INC)
list(GET JL_OUT 5 CC)
list(GET JL_OUT 6 CXX)
list(GET JL_OUT 7 CMAKE)
list(GET JL_OUT 8 MAKE)
list(GET JL_OUT 9 GIT)
list(GET JL_OUT 10 PYTHON)
list(GET JL_OUT 11 TF_LIB_FILE)

message("Python path=${PYTHON}")
message("EIGEN_INC=${EIGEN_INC}")
message("TF_INC=${TF_INC}")
message("TF_ABI=${TF_ABI}")
message("TF_LIB_FILE=${TF_LIB_FILE}")

# https://github.com/tensorflow/tensorflow/issues/1569
# if GCC > 5
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${TF_ABI} ${CMAKE_CXX_FLAGS}")
endif()

include_directories(${TF_INC} ${EIGEN_INC})


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")


add_library(SparseMatMul SHARED SparseMatMul.cpp)
set_property(TARGET SparseMatMul PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(SparseMatMul ${TF_LIB_FILE})
