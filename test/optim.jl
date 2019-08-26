@testset "NLopt" begin
######################### integration with NLopt.jl #########################

# ================ standard NLopt ================
function myfunc(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 0
        grad[2] = 2x[2]
    end
    return x[2]^2
end

function myconstraint(x::Vector, grad::Vector, a, b)
    if length(grad) > 0
        grad[1] = 2*(x[1]-1)
        grad[2] = -1
    end
    (x[1]-1)^2 - x[2]
end

opt = Opt(:LD_MMA, 2)
opt.lower_bounds = [-Inf, 0.0]
opt.xtol_rel = 1e-4

opt.min_objective = myfunc
inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 1e-8)

(minf,minx,ret) = NLopt.optimize(opt, [1.234, 5.678])
# ================ END NLopt ================
p = ones(10)
Con = CustomOptimizer() do f, df, c, dc, x0, args...
    opt = Opt(:LD_MMA, length(x0))
    bd = zeros(length(x0)); bd[end-1:end] = [-Inf, 0.0]
    opt.lower_bounds = bd
    opt.xtol_rel = 1e-4
    opt.min_objective = (x,g)->(g[:]= df(x); return f(x)[1])
    inequality_constraint!(opt, (x,g)->( g[:]= dc(x);c(x)[1]), 1e-8)
    (minf,minx,ret) = NLopt.optimize(opt, x0)
    minx
end

# unfortunately, we must ensure only minimization variables are in the current tensorflow graph
# for NLopt to work properly
reset_default_graph() 
x = Variable([1.234; 5.678])
y = Variable([1.0;2.0])
loss = x[2]^2 + sum(y^2)
c1 = (x[1]-1)^2 - x[2] 
opt = Con(loss, inequalities=[c1],var_to_bounds=Dict(x=>[-1.,1.]))
sess = Session(); init(sess)
opt.minimize(sess)
xmin = run(sess, x)

@test true
end


@testset "Optim" begin
######################### integration with Optim.jl #########################

NonCon = CustomOptimizer() do f, df, c, dc, x0, args...
    # @show f, df, c, dc, x0
    res = Optim.optimize(f, df, x0; inplace = false)        
    res.minimizer
end
reset_default_graph()
x = Variable(rand(2))
f = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
f1(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
res = Optim.optimize(f1, rand(2))
opt = NonCon(f)
sess = Session(); init(sess)
opt.minimize(sess)
xmin = run(sess, x)
@test norm(xmin-res.minimizer)<1e-2

end


#=
@testset "Ipopt" begin
######################### integration with Ipopt.jl #########################
# https://github.com/jainachin/tf-ipopt/blob/master/examples/hs071.py
IPOPT = CustomOptimizer() do f, df, c, dc, x0, nineq, neq, x_L, x_U
    n = length(x0)
    nz = length(dc(x0))
    g_L, g_U = [-2e19;0.0], [0.0;0.0]
    function eval_jac_g(x, mode, rows, cols, values)
        if mode == :Structure
            rows[1] = 1; cols[1] = 1
            rows[2] = 1; cols[2] = 1
            rows[3] = 2; cols[3] = 1
            rows[4] = 2; cols[4] = 1
        else
            values[:]=dc(x)
        end
    end
    prob = Ipopt.createProblem(n, x_L, x_U, 2, g_L, g_U, nz, 0,
            f, (x,g)->(g[:]=c(x)), (x,g)->(g[:]=df(x)), eval_jac_g, nothing)
    addOption(prob, "hessian_approximation", "limited-memory")

    prob.x = x0
    status = Ipopt.solveProblem(prob)
    println(Ipopt.ApplicationReturnStatus[status])
    println(prob.x)
    prob.x
end

reset_default_graph()
x = Variable([1.0;1.0])
x1 = x[1]; x2 = x[2]; 
loss = x2
g = x1
h = x1*x1 + x2*x2 - 1
opt = IPOPT(loss, inequalities=[g], equalities=[h], var_to_bounds=Dict(x=>(-1.0,1.0)))
sess = Session(); init(sess)
opt.minimize(sess)
@test true
=#