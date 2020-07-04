cmake_minimum_required(VERSION 3.5)
project(MPI)


find_package(MPI)

# set mpi if not found
if (MPI_FOUND)
else()
set(MPI_INCLUDE_PATH /usr/lib64/mpich2/include)
set(MPI_C_LIBRARIES /usr/lib64/openmpi/lib/libmpi.so)
endif (MPI_FOUND)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
add_library(mtest SHARED mpitest.cpp)
target_link_libraries(mtest ${MPI_C_LIBRARIES})