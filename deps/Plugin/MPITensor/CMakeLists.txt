cmake_minimum_required(VERSION 3.5)
project(TF_CUSTOM_OP)
set (CMAKE_CXX_STANDARD 11)

message("JULIA=${JULIA}")

IF(JULIA)
ELSE()
find_program(JULIA "julia")
ENDIF()

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
list(GET JL_OUT 12 LIBCUDA)
list(GET JL_OUT 13 CUDA_INC)

message("Python path=${PYTHON}")
message("PREFIXDIR=${PREFIXDIR}")
message("TF_INC=${TF_INC}")
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

if(MSVC)
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
  endif()
  add_definitions(-DNOMINMAX)
endif()

include_directories(${TF_INC} ${PREFIXDIR} ${CUDA_INC} ${LIBDIR}/../include)
link_directories(${LIBDIR})


# Compile MPI programs
IF(DEFINED ENV{MPI_C_LIBRARIES})
  set(MPI_INCLUDE_PATH $ENV{MPI_INCLUDE_PATH})
  set(MPI_C_LIBRARIES $ENV{MPI_C_LIBRARIES})
  message("MPI_INCLUDE_PATH = ${MPI_INCLUDE_PATH}")
  message("MPI_C_LIBRARIES = ${MPI_C_LIBRARIES}")
  include_directories(${MPI_INCLUDE_PATH})
ELSE()
  message("MPI_INCLUDE_PATH and/or MPI_C_LIBRARIES is not set. MPI operators are not compiled.")
ENDIF()

find_library(hypre HYPRE HINTS ${LIBDIR})
message("hypre: ${hypre}")

add_library(MPITensor SHARED MPITensor.cpp MPITensorTranspose.cpp)

set_property(TARGET MPITensor PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(MPITensor ${TF_LIB_FILE} ${hypre})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set_target_properties(MPITensor PROPERTIES 
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build 
                RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/build
                ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/build)
