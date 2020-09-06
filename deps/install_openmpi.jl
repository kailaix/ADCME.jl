using ADCME

CONDA_ROOT = abspath(joinpath(ADCME.LIBDIR, ".."))
change_directory(ADCME.PREFIXDIR)
http_file("https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.4.tar.gz", "openmpi.tar.gz")
uncompress("openmpi.tar.gz", "openmpi-4.0.4")
change_directory(joinpath("openmpi-4.0.4", "build"))
require_file("Makefile") do
    run_with_env(`../configure CC=$(ADCME.CC) CXX=$(ADCME.CXX) --enable-mpi-thread-multiple --prefix=$(CONDA_ROOT) 
                    --enable-mpirun-prefix-by-default --enable-mpi-fortran=no --with-mpi-param-check=always`, Dict("LDFLAGS"=>"-L"*ADCME.LIBDIR))
end
require_library("mpi") do 
    run_with_env(`make -j all`)
end
require_library(joinpath(ADCME.LIBDIR, "libmpi")) do 
    run_with_env(`make install`)
end
