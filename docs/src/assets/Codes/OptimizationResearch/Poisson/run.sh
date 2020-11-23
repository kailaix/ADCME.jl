ulimit -u 10000
for SEED in 2 23 233 2333 23333
do 
julia bfgs.jl $SEED &
done 
wait 