
export install_adept, install_blas, install_openmpi

function install_blas(blas_binary)
    if Sys.iswindows()
        if isfile(joinpath(ADCME.LIBDIR, "openblas.lib"))
            return 
        end 
        if blas_binary
            @info "Downloading prebuilt blas from Github. If you encounter any problem with openblas when using adept, run `install_adept(blas_binary=false)` to compile from source"
            download("https://github.com/kailaix/tensorflow-1.15-include/releases/download/v0.1.0/openblas.lib", joinpath(ADCME.LIBDIR, "openblas.lib"))
            return 
        end
        @info "You are building openblas from source on Windows, and this process may take a long time.
Alternatively, you can place your precompiled binary to $(joinpath(ADCME.LIBDIR, "openblas.lib"))"
        PWD = pwd()
        download("https://github.com/xianyi/OpenBLAS/archive/v0.3.9.zip", joinpath(ADCME.LIBDIR, "OpenBlas.zip"))
        cd(ADCME.LIBDIR)
        run(`cmd /c unzip OpenBLAS.zip`)
        rm("OpenBlas", force=true, recursive=true)
        run(`cmd /c ren OpenBlas-0.3.9 OpenBlas`)
        rm("OpenBlas.zip")
        cd("OpenBlas")
        mkdir("build")
        cd("build")
        ADCME.cmake(CMAKE_ARGS="-DCMAKE_Fortran_COMPILER=flang -DBUILD_WITHOUT_LAPACK=no -DNOFORTRAN=0 -DDYNAMIC_ARCH=ON")
        ADCME.make()
        cd("../build/lib/Release")
        mv("openblas.lib", joinpath(ADCME.LIBDIR, "openblas.lib"))
        cd(PWD)
    else 
        required_file = Sys.isapple() ? ".dylib" : ".so"
        required_file = joinpath(ADCME.LIBDIR, "libopenblas")*required_file
        if !isfile(required_file)
            files = readdir(ADCME.LIBDIR)
            files = filter(x->!isnothing(x), match.(r"(libopenblas\S*.dylib)", files))[1]
            target = joinpath(ADCME.LIBDIR, files[1])
            symlink(target, required_file)
            @info "Symlink $(required_file) --> $(files[1])"
        end
    end 
end

"""
    install_adept(force::Bool=false)

Install adept-2 library: https://github.com/rjhogan/Adept-2
"""
function install_adept(force::Bool=false; blas_binary::Bool = true)
    PWD = pwd()
    cd(ADCME.LIBDIR)
    if force 
        @info "Removing Adept-2 by force..."
        rm("Adept-2", force=true, recursive=true)
    end
    if !isdir("Adept-2") 
        LibGit2.clone("https://github.com/ADCMEMarket/Adept-2", "Adept-2")
    end
    cd("Adept-2/adept")
    install_blas(blas_binary)
    try
        if (!isfile("$(LIBDIR)/libadept.so") && !isfile("$(LIBDIR)/libadept.dylib") && !isfile("$(LIBDIR)/adept.lib")) || force
            @info """Copy "$(@__DIR__)/../deps/AdeptCMakeLists.txt" to "$(joinpath(pwd(), "CMakeLists.txt"))" ... """
            cp("$(@__DIR__)/../deps/AdeptCMakeLists.txt", "./CMakeLists.txt", force=true)
            @info """Remove $(joinpath(pwd(), "build")) ... """
            rm("build", force=true, recursive=true)
            @info "Make $(joinpath(pwd(), "build")) ... "
            mkdir("build")
            @info "Change directory into $(joinpath(pwd(), "build")) ... "
            cd("build")
            @info "Cmake ... "
            ADCME.cmake()
            @info "Make ... "
            ADCME.make()
        end
        printstyled("""
∘ Add the following lines to CMakeLists.txt 

include_directories(\${LIBDIR}/Adept-2/include)
find_library(ADEPT_LIB_FILE adept HINTS \${LIBDIR})
find_library(LIBOPENBLAS openblas HINTS \${LIBDIR})
message("ADEPT_LIB_FILE=\${ADEPT_LIB_FILE}")
message("LIBOPENBLAS=\${LIBOPENBLAS}")

∘ Add `\${ADEPT_LIB_FILE}` and `\${LIBOPENBLAS}` to `target_link_libraries`
""", color=:green)
    catch
        printstyled("Compliation failed\n", color=:red)
    finally
        cd(PWD)
    end
end



CONDA_ROOT = abspath(joinpath(LIBDIR, ".."))
function maybe_download(URL)
    filename = splitdir(URL)[2]
    file = joinpath(PREFIXDIR, filename)
    if ispath(file)
        return file
    end
    download(URL, file)
    return file 
end

function maybe_tar_gz(file)
    filename = file[1:end-7]
    if isdir(filename)
        return 
    else
        run(`tar zxvf $file`)
    end
end

function install_openmpi()
    if Sys.iswindows()
        @warn "OpenMPI is not fully supported on Windows. Please install Microsoft MPI: https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi"
    end
    PWD = pwd()
    cd(PREFIXDIR)
    file = maybe_download("https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.4.tar.gz")
    maybe_tar_gz(file)
    cd("openmpi-4.0.4")
    if !isdir("build")
        mkdir("build")
    end
    cd("build")
    run(`../configure CC=$CC CXX=$CXX --enable-mpi-thread-multiple --prefix=$CONDA_ROOT --enable-mpirun-prefix-by-default --with-mpi-param-check=always`)
    run(`$MAKE -j all`)
    run(`$MAKE install`)
    printstyled("""Please add MPI_INCLUDE_PATH and MPI_C_LIBRARIES to your environment path
Linux:
export MPI_INCLUDE_PATH=$INCDIR
export MPI_C_LIBRARIES=$(joinpath(LIBDIR, "libmpi.so"))
MacOS:
export MPI_INCLUDE_PATH=$INCDIR
export MPI_C_LIBRARIES=$(joinpath(LIBDIR, "libmpi.dylib"))
""", color=:green)
    cd(PWD)
end
