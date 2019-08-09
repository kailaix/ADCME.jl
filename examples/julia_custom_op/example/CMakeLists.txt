cmake_minimum_required(VERSION 3.5)
project(TFApp)

# Julia utilities
execute_process(COMMAND julia -e "abspath(joinpath(Base.find_package(\"ADCME\"), \"../../examples/custom_op/headers\"))|>print" OUTPUT_VARIABLE HEADERS)
execute_process(COMMAND julia -e "abspath(joinpath(Sys.BINDIR, \"../include/julia\"))|>print" OUTPUT_VARIABLE JL_INC)
execute_process(COMMAND julia -e "abspath(joinpath(Sys.BINDIR, \"../lib\"))|>print" OUTPUT_VARIABLE JL_LIB)
execute_process(COMMAND julia -e "abspath(joinpath(Sys.BINDIR, \"../lib/julia\"))|>print" OUTPUT_VARIABLE JL_LIB_INC)
include_directories(${HEADERS} ${JL_INC})
link_directories(${JL_LIB} ${JL_LIB_INC})
add_definitions(-DJULIA_ENABLE_THREADING)

set (CMAKE_CXX_STANDARD 11)
# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

add_executable(TwoLayer CustomOp.cpp)
# set_property(TARGET TwoLayer PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(TwoLayer libjulia.1.1.dylib ${CMAKE_BINARY_DIR}/JuliaModule.dylib)
