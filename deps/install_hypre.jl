using ADCME

PWD = pwd()
change_directory()
http_file("https://github.com/kailaix/hypre/archive/v2.19.0.tar.gz", "v2.19.0.tar.gz")
uncompress("v2.19.0.tar.gz", "hypre-2.19.0")
change_directory("hypre-2.19.0/src/build")
ROOT = joinpath(ADCME.BINDIR, "..")
run_with_env(`$(ADCME.CMAKE) -G Ninja -DCMAKE_MAKE_PROGRAM=$(ADCME.NINJA)
            -DHYPRE_SHARED:BOOL=ON -DHYPRE_INSTALL_PREFIX:PATH=$ROOT 
            -DCMAKE_C_COMPILER:FILEPATH=$(ADCME.CC) -DCMAKE_CXX_COMPILER:FILEPATH=$(ADCME.CXX) ..`)
run_with_env(`$(ADCME.CMAKE) -G Ninja -DCMAKE_MAKE_PROGRAM=$(ADCME.NINJA) -L ..`)
ADCME.make()
run_with_env(`$(ADCME.NINJA) install`)
# run_with_env(`mv $(ROOT)/lib64/libHYPRE.so $(ADCME.LIBDIR)/libHYPRE.so`)
cd(PWD)