# julia bfgs_with_adam_hessian.jl &
# julia bfgs_with_adam_nohessian.jl &
# julia bfgs_with_noadam.jl &
# julia adam.jl &
# julia lbfgs_with_adam.jl &
# julia lbfgs_with_noadam.jl &
# wait 

ulimit -u 10000
for N in 100 200
do 
# julia bfgs_with_adam_hessian.jl $N &
# julia bfgs_with_adam_nohessian.jl $N &
# julia lbfgs_with_adam.jl $N &
done 
wait 

