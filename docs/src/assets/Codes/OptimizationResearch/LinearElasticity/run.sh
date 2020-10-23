ulimit -u 10000
julia adam.jl &
julia bfgs_adam_hessian.jl &
julia bfgs_adam_nohessian.jl &
julia bfgs.jl &
julia lbfgs.jl &
julia lbfgs_adam.jl &
wait 