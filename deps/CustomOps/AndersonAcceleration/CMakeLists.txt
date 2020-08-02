cmake_minimum_required(VERSION 3.5)
project(Adept LANGUAGES C CXX)

set (CMAKE_CXX_STANDARD 11)

message("JULIA=${JULIA}")
execute_process(COMMAND ${JULIA} -e "import ADCME; print(ADCME.__STR__)" OUTPUT_VARIABLE JL_OUT)
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

set(CMAKE_BUILD_TYPE Release)
if(MSVC)
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG")
else()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

if(MSVC)
find_library(LIBOPENBLAS libblas HINTS ${LIBDIR})
find_library(LIBLAPACK liblapack HINTS ${LIBDIR})
else()
find_library(LIBOPENBLAS openblas HINTS ${LIBDIR})
find_library(LIBLAPACK lapack HINTS ${LIBDIR})
endif()

message("${LIBLAPACK}")
add_library(AA SHARED libaa.cpp aa.c)
set_property(TARGET AA PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(AA ${LIBOPENBLAS} ${LIBLAPACK})

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set_target_properties(AA PROPERTIES 
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build 
                RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/build
                ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/build)
