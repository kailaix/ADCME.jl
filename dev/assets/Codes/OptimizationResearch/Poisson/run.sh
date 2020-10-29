ulimit -u 10000

for seed in 1 
do 
    # julia bfgs_with_adam_hessian.jl $seed &
    # julia bfgs_with_adam_nohessian.jl $seed &
    # julia bfgs_with_noadam.jl $seed &
    # julia adam.jl $seed &
    # julia lbfgs_with_adam.jl $seed &
    # julia lbfgs_with_noadam.jl $seed &
    julia damped_bfgs.jl $seed &
    julia damped_bfgs_with_adam_nohessian.jl $seed &
    wait 
done 
