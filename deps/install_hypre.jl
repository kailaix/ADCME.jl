using BinDeps
using ADCME
using CMake

@BinDeps.setup 

const libhypre = library_dependency("libhypre", aliases=["libhypre", "libHYPRE"])
const hyprever = "2.19.0"
provides(Sources, 
    URI("https://github.com/kailaix/hypre/archive/v$hyprever.tar.gz"),
    libhypre, unpacked_dir = "hypre-$hyprever")

libhypre.context.dir = ADCME.PREFIXDIR
cmakedir = joinpath(srcdir(libhypre), "hypre-$hyprever", "src", "build")
cc = ADCME.CC
cxx = ADCME.CXX
CONDA_ROOT = joinpath(ADCME.BINDIR, "..")

provides(SimpleBuild,
(@build_steps begin
    GetSources(libhypre)
    CreateDirectory(cmakedir)
    @build_steps begin
        ChangeDirectory(cmakedir)
        `$cmake -DHYPRE_SHARED:BOOL=ON -DHYPRE_INSTALL_PREFIX:PATH=$CONDA_ROOT -DCMAKE_C_COMPILER:FILEPATH=$(cc) -DCMAKE_CXX_COMPILER:FILEPATH=$(cxx) ..`
        `$cmake -L ..`
        MakeTargets(".", ["all"])
        MakeTargets(".", ["install"])
        FileRule(joinpath(ADCME.LIBDIR, "libHYPRE.so"),
            @build_steps begin
                `ln -s $(CONDA_ROOT)/lib64/libHYPRE.so $(ADCME.LIBDIR)/libHYPRE.so`
            end
        )
    end
end),
libhypre,
os = :Linux)

BinDeps.execute(libhypre, BinDeps.SimpleBuild)


