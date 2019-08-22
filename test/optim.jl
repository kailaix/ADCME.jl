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

Con = CustomOptimizer() do f, df, c, dc, x0, nineq, neq
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
loss = x[2]^2
c1 = (x[1]-1)^2 - x[2]
opt = Con(loss, inequalities=[c1])
sess = Session(); init(sess)
opt.minimize(sess)
xmin = run(sess, x)
@test norm(xmin-minx)≈0.0

end


@testset "Optim" begin
######################### integration with Optim.jl #########################

NonCon = CustomOptimizer() do f, df, c, dc, x0, nineq, neq
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
######################### integration with Ipopt.jl #########################
# https://github.com/jainachin/tf-ipopt/blob/master/examples/hs071.py
CustomOptimizer("IPOPT") do f, df, c, dc, x0, nineq, neq
    @show "here"
    error()
    n = length(x0)
    m = 2
    x_L = [1.0, 1.0, 1.0, 1.0]
    x_U = [5.0, 5.0, 5.0, 5.0]
    g_L, g_U = [-2e19;0.0], [0.0;0.0]
    eval_jac_g = (x, mode, rows, cols, values) -> (values[:]=dc(x))
    eval_h = (x, mode, rows, cols, obj_factor, lambda, values)->(values[:].=0.0)
    @show "here"
    prob = Ipopt.createProblem(n, x_L, x_U, m, g_L, g_U, 8, 0,
            f, (x,g)->(g[:]=c(x)), (x,g)->(g[:]=df(x)), eval_jac_g, eval_h)
    prob.x = x0
    status = Ipopt.solveProblem(prob)
    println(Ipopt.ApplicationReturnStatus[status])
    println(prob.x)
    prob.x
end

reset_default_graph()
x1, x2, x3, x4 = Variable(1.0), Variable(5.0), Variable(5.0), Variable(1.0)
loss = x1 * x4 * (x1 + x2 + x3) + x3
g = (x1 * x2 * x3 * x4 - 25)
h = x1*x1 + x2*x2 + x3*x3 + x4*x4 - 40
opt = IPOPT(loss, inequalities=[g], equalities=[h])
sess = Session(); init(sess)
opt.minimize(loss)
=#
