cmake_minimum_required(VERSION 3.5)
project(TF_CUSTOM_OP)
set (CMAKE_CXX_STANDARD 11)

execute_process(COMMAND ${JULIA} -e "using Conda; print(joinpath(Conda.LIBDIR, \"Libraries\"))" OUTPUT_VARIABLE PREFIXDIR)
execute_process(COMMAND ${JULIA} -e "using Conda; print(joinpath(Conda.PYTHONDIR, \"python\"))" OUTPUT_VARIABLE PYTHON)
execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; import sys; \
                                       TF_INC = tf.sysconfig.get_compile_flags()[0][2:];\
                                       TF_LIB = tf.sysconfig.get_link_flags()[0][2:];\
                                       TF_ABI = tf.sysconfig.get_compile_flags()[1][-1];\
                                       TF_LIB_FILE = tf.sysconfig.get_link_flags()[1][3:];\
                                       OUT = ';'.join([TF_INC,TF_LIB,TF_ABI,TF_LIB_FILE]);\
                                       sys.stdout.write(OUT)" OUTPUT_VARIABLE PY_OUT)
list(GET PY_OUT 0 TF_INC)
list(GET PY_OUT 1 TF_LIB)
list(GET PY_OUT 2 TF_ABI)
list(GET PY_OUT 3 TF_LIB_FILE)

message("Python path=${PYTHON}")
message("PREFIXDIR=${PREFIXDIR}")
message("TF_INC=${TF_INC}")
message("TF_LIB=${TF_LIB}")
message("TF_ABI=${TF_ABI}")
message("TF_LIB_FILE=${TF_LIB_FILE}")


if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${TF_ABI} ${CMAKE_CXX_FLAGS}")
endif()

set(CMAKE_BUILD_TYPE Release)
if(MSVC)
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG")
else()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

include_directories(${TF_INC} ${PREFIXDIR}/../../include ${PREFIXDIR} ${TORCH_INC})
link_directories(${TF_LIB})

find_package(CUDA QUIET)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-shared ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-fPIC ${CMAKE_CXX_FLAGS}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr)
SET(CUDA_PROPAGATE_HOST_FLAGS ON)

find_program(_nvidia_smi "nvidia-smi")
if (_nvidia_smi)
  add_definitions(-DUSE_GPU)
  message("Compiling GPU-compatible custom operator!")
  cuda_add_library(MyAssign SHARED MyAssign.cpp MyAssign.cu)
else()
  add_library(MyAssign SHARED MyAssign.cpp)
endif()

set_property(TARGET MyAssign PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(MyAssign ${TF_LIB_FILE} "${TORCH_LIBRARIES}")
