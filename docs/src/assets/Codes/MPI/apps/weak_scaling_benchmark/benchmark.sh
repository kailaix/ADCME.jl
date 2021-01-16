# salloc --nodes=20

for n in 1 4 9 16 25 36 49 64
do 
    mpirun -n $n --map-by node:pe=1 --report-bindings julia benchmark.jl 1
done 

for n in 1 4 9 16 
do 
    mpirun -n $n --map-by node:pe=4 --report-bindings julia benchmark.jl 4
done 

