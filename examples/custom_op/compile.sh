cd ADEL
rm -rf build
mkdir build
cd build
cmake ..
set -x
make -j 
cd ..
cd ..

cd inner_product
rm -rf build
mkdir build
cd build
cmake ..
make -j 
cd ..
cd ..

cd nonlinear
rm -rf build
mkdir build
cd build
cmake ..
make -j 
cd ..
cd ..