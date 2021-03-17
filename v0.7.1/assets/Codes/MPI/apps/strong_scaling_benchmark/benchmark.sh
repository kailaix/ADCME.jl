# salloc --nodes=20

for n in  25 16 9 4 1
do 
    /home/kailaix/.julia/adcme/bin/mpirun -n $n --map-by slot:pe=1 --report-bindings julia benchmark.jl $n 1
done 

for n in 16 9 4 1
do 
    /home/kailaix/.julia/adcme/bin/mpirun -n $n --map-by slot:pe=2 --report-bindings julia benchmark.jl $n 2
done 

for n in 9 4 1
do 
    /home/kailaix/.julia/adcme/bin/mpirun -n $n --map-by slot:pe=3 --report-bindings julia benchmark.jl $n 3
done 
