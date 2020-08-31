# Built-in Toolchain for Third-party Libraries

Using third-party libraries, especially binaries with operation system and runtime dependencies, is troublesome and painful. ADCME has a built-in set of tools for downloading, uncompressing, and compiling third-party libraries. Therefore, we can compile external libraries from source and ensure that the products are compatible within the ADCME ecosystem. 

The toolchain uses the same set of C/C++ compilers that were used to compile the tensorflow dynamic library. Additionally, the toolchain provide essential building  tools such as `cmake` and `ninja`. This section will be a short introduction to the toolchain. 

The best way to introduce the toolchain is through an example. Let us consider compiling a C++ library for use in ADCME. The library [LibHelloWorld](https://github.com/kailaix/LibHelloWorld) is hosted on GitHub and the repository contains a `CMakeLists.txt` and `HelloWorld.cpp`. Our goal is to download the repository into ADCME private workspace (`ADCME.PREFIXDIR`) and compile the library to the library path (`ADCME.LIBDIR`). The following code will do the job:


```julia
using ADCME

PWD = pwd()
change_directory(ADCME.PREFIXDIR)
git_repository("https://github.com/kailaix/LibHelloWorld", "LibHelloWorld")
change_directory("LibHelloWorld")
make_directory("build")
change_directory("build")
lib = get_library("hello_world")
require_file(lib) do 
    ADCME.cmake()
    ADCME.make()
end
_, libname = splitdir(lib)
mv(lib, joinpath(ADCME.LIBDIR, libname), force=true)
change_directory(PWD)
```

Here [`change_directory`](@ref), [`git_repository`](@ref), and others are ADCME toolchain functions. Note we have used `ADCME.cmake()` and `ADCME.make()` to ensure that the codes are compiled with a compatible compiler. The toolchain will cache all the intermediate files and therefore will not recompile as long as the files exist. To force recompiling, users need to delete the local repository, i.e., `LibHelloWorld` in `ADCME.PREFIXDIR`. The following is an exemplary output of the program:

```bash
[ Info: Changed to directory /home/darve/kailaix/.julia/adcme/lib/Libraries
[ Info: Cloning from https://github.com/kailaix/LibHelloWorld to LibHelloWorld...
[ Info: Cloned https://github.com/kailaix/LibHelloWorld to LibHelloWorld
[ Info: Changed to directory LibHelloWorld
[ Info: Made directory directory
[ Info: Changed to directory build
-- The C compiler identification is GNU 5.4.0
-- The CXX compiler identification is GNU 5.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /home/darve/kailaix/.julia/adcme/bin/x86_64-conda_cos6-linux-gnu-gcc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /home/darve/kailaix/.julia/adcme/bin/x86_64-conda_cos6-linux-gnu-g++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
JULIA=/home/darve/kailaix/julia-1.3.1/bin/julia
...
Python path=/home/darve/kailaix/.julia/adcme/bin/python
PREFIXDIR=/home/darve/kailaix/.julia/adcme/lib/Libraries
TF_INC=/home/darve/kailaix/.julia/adcme/lib/python3.7/site-packages/tensorflow_core/include
TF_ABI=1
TF_LIB_FILE=/home/darve/kailaix/.julia/adcme/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so.1
-- Configuring done
-- Generating done
-- Build files have been written to: /home/darve/kailaix/.julia/adcme/lib/Libraries/LibHelloWorld/build
[2/2] Linking CXX shared library libhello_world.so
[ Info: Changed to directory /home/darve/kailaix/project/MPI_Project/LibHelloWorld
```

To use the compiled library, we can write 

```julia
using ADCME
libdir = joinpath(ADCME.LIBDIR, "libhello_world.so")
@eval ccall((:helloworld, $libdir), Cvoid, ())
```

Then we get the expected output: 

```bash
Hello World!
```

The design idea for ADCME toolchains is that users can write an install script. Then ADCME will guarantee that the runtime and compilation eco-system are compatible. 