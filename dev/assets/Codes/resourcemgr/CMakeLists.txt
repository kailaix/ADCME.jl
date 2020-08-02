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

add_library(TestResourceManager SHARED TestResourceManager.cpp)

set_property(TARGET TestResourceManager PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(TestResourceManager ${TF_LIB_FILE} "${TORCH_LIBRARIES}")
