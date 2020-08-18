using BinDeps
using ADCME

@BinDeps.setup 

libmpi = library_dependency("libmpi")
provides(Sources, 
    URI("https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.4.tar.gz"),
    libmpi)
libmpi.context.dir = ADCME.PREFIXDIR
CONDA_ROOT = joinpath(ADCME.BINDIR, "..")
cmakebuilddir = joinpath(srcdir(libmpi), "openmpi-4.0.4", "build")
cc = ADCME.CC
cxx = ADCME.CXX
CONDA_ROOT = joinpath(ADCME.BINDIR, "..")
provides(SimpleBuild, (@build_steps begin
    GetSources(libmpi)
    CreateDirectory(cmakebuilddir)
    @build_steps begin
        ChangeDirectory(cmakebuilddir)
        `../configure CC=$cc CXX=$cxx --enable-mpi-thread-multiple --prefix=$CONDA_ROOT --enable-mpirun-prefix-by-default --with-mpi-param-check=always`
        `make -j all`
        `make install`
    end
end),
libmpi, os=:Unix)
BinDeps.execute(libmpi, BinDeps.SimpleBuild)