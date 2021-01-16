# salloc --nodes=20

for n in  64 25 16 4 1
do 
    /home/kailaix/.julia/adcme/bin/mpirun -n $n --map-by node:pe=1 --report-bindings julia benchmark.jl $n 1
done 

for n in 16 4 1
do 
    /home/kailaix/.julia/adcme/bin/mpirun -n $n --map-by node:pe=4 --report-bindings julia benchmark.jl $n 4
done 

