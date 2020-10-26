# Configure MPI for Distributed Computing

This section will cover how to configure ADCME for MPI functionalities.

## Configure the MPI backend 

The first step is to configure your MPI backend. There are many choices depending on your operation system. For example, Windows have Microsoft MPI. There are also OpenMPI and Intel MPI available on most Linux distributions. If you want to use your own MPI backend, you need to locate the MPI libraries, header files, and executable (e.g., `mpirun`). You need to build ADCME with the following environment variable:

* `MPI_C_LIBRARIES`: the MPI shared library, for example, on Windows, it may be 
```
C:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Lib\\x64\\msmpi.lib
```
On Unix systems, it may be 
```/opt/ohpc/pub/compiler/intel-18/compilers_and_libraries_2018.2.199/linux/mpi/intel64/lib/release/libmpi.so```
Note that you must include **the shared library** in the variable. 
* `MPI_INCLUDE_PATH`: the directory where `mpi.h` is located, for example,
```C:\\Program Files (x86)\\Microsoft SDKs\\MPI\\Include```
Or in a Unix system, we have 
```/opt/ohpc/pub/compiler/intel-18/compilers_and_libraries_2018.2.199/linux/mpi/intel64/include/```

The simplest way is to add these variables in the environment variables. For example, in Linux, we can add the following lines in the `~/.bashrc` file. 
```bash
export MPI_C_LIBRARIES=/opt/ohpc/pub/compiler/intel-18/compilers_and_libraries_2018.2.199/linux/mpi/intel64/lib/libmpi.so
export MPI_INCLUDE_PATH=/opt/ohpc/pub/compiler/intel-18/compilers_and_libraries_2018.2.199/linux/mpi/intel64/include/
alias mpirun=/opt/ohpc/pub/compiler/intel-18/compilers_and_libraries_2018.2.199/linux/mpi/intel64/bin/mpirun
```

In the case you do not have an MPI backend, ADCME provides you a convenient way to install MPI by compiling from source. Just run 
```julia
using ADCME
install_openmpi()
```
This should install an OpenMPI library for you. Note this functionality does not work on Windows and is only tested on Linux. 

## Build MPI Libraries 

The MPI functionality of ADCME is not fulfilled at this point. To enable the MPI support, you need to recompile the built-in custom operators.
```julia
using ADCME
ADCME.precompile(true)
```
At this point, you will be able to use MPI features. 

## Build MPI Custom Operators

You can also build MPI-enabled custom operators by calling
```julia
using ADCME
customop(with_mpi=true)
```

In this case, there will be extra lines in `CMakeLists.txt` to setup MPI dependencies.
```julia
IF(DEFINED ENV{MPI_C_LIBRARIES})
  set(MPI_INCLUDE_PATH $ENV{MPI_INCLUDE_PATH})
  set(MPI_C_LIBRARIES $ENV{MPI_C_LIBRARIES})
  message("MPI_INCLUDE_PATH = ${MPI_INCLUDE_PATH}")
  message("MPI_C_LIBRARIES = ${MPI_C_LIBRARIES}")
  include_directories(${MPI_INCLUDE_PATH})
ELSE()
  message("MPI_INCLUDE_PATH and/or MPI_C_LIBRARIES is not set. MPI operators are not compiled.")
ENDIF()
```

## Running MPI Applications with Slurm

To run MPI applications with slurm, the following commands are useful

```bash
sbatch -n 4 -c 8 mpirun -n 4 julia app.jl 
```
This specifies 4 tasks and each task uses 8 cores. You can also replace `sbatch` with `salloc`.

To diagonose the application, you can also let `mpirun` print out the rank information, e.g., in OpenMPI we have

```bash
sbatch -n 4 -c 8 mpirun --report-bindings -n 4 julia app.jl
```

