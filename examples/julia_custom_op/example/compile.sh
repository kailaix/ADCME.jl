set -v
juliac.jl -vasi JuliaModule.jl
mkdir build
cp builddir/*.dylib build/  # The new library/executable must be companied by julia library next to it (@rpath issue?)
cd build
cmake ..
make -j 
./TwoLayer