


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
        @show f, df, c, dc, x0
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

@testset "newton raphson" begin
@test_skip begin

    A = rand(10,10)
    A = A'*A + I
    rhs = rand(10)
    function newton_raphson_f(θ, u)
        r = A*u - rhs
        r, constant(A)
    end
    ADCME.options.newton_raphson.verbose = true
    nr = newton_raphson(newton_raphson_f, zeros(10), missing)
    nr = run(sess, nr)
    @test nr.x≈A\rhs

    u0 = rand(10)
    function newton_raphson_f2(θ, u)
        r = u^3+u - u0
        r, spdiag(3u^2+1.0)
    end

    ADCME.options.newton_raphson.verbose = true
    ADCME.options.newton_raphson.tol = 1e-5
    nr = newton_raphson(newton_raphson_f2, rand(10), missing)
    nr = run(sess, nr)
    uval = nr.x
    @test norm(uval.^3+uval-u0)<1e-3

    # least square
    u0 = rand(10)
    rs = rand(10)
    function newton_raphson_f3(θ, u)
        r = [u^2;u] - [rs.^2;rs]
        r, [spdiag(2*u); spdiag(10)]
    end
    nr = newton_raphson(newton_raphson_f3, rand(10), missing)
    nr = run(sess, nr)
    uval = nr.x
    @test norm(uval-rs)<1e-3
end
end

@testset "NonlinearConstrainedProblem" begin
    @test_skip begin
    θ = Variable(1.8*ones(3))
    u0 = ones(3)
    function f1(θ, u)
        r = u+u^3-θ
        A = spdiag(3u^2+1.0)
        r, A
    end
    function L1(u)
        return sum((u-u0)^2)
    end
    l, u, g = NonlinearConstrainedProblem(f1,L1,θ,zeros(3))
    # the negative gradients and vars are (-g, θ)
    opt_ = AdamOptimizer(1e-1).apply_gradients([(g, θ)])
    init(sess)
    for i = 1:500
        run(sess, opt_)
    end
    @test norm(run(sess, θ)-2.0*ones(3))<1e-8


    θ = constant(ones(1))
    u0 = ones(3)
    Random.seed!(233)
    A = round.(rand(3, 3), digits=1)
    t = zeros(3); t[1:2].=1.0
    function f1(θ, u)
        # r = u-sum(θ)^2-t*sum(θ)
        # A = spdiag(length(u))
        # r, A
        # -t*sum(θ)
        A*u-[sum(θ);sum(θ);constant(0.0)], constant(A)
    end
    function L1(u)
        return sum(u)
    end
    close("all")

    function value_and_gradients_function(θ)
        l, u, g = NonlinearConstrainedProblem(f1,L1,θ,zeros(3))
        l, g
    end
    results = BFGS!(value_and_gradients_function, zeros(3))
    u = run(sess, results)
    @test u≈[2.;2.;2.]
    end
end

@testset "Custom BFGS!" begin
    reset_default_graph()
    x = Variable(2ones(10))
    z = Variable(1.0)
    w = Variable(ones(20,30,10))
    loss = sum((x-1.0)^2+z^2+w^2)
    grads = [gradients(loss, x), gradients(loss, z), gradients(loss, w)]
    vars = [x,z, w]
    global sess = Session(); init(sess)
    l = BFGS!(sess, loss, grads, vars)
    @test l[end]<1e-5
end

@testset "var_to_bounds" begin 
    x = Variable(2.0)    
    loss = x^2
    init(sess)
    BFGS!(sess, loss, bounds=Dict(x=>[1.0,3.0]))
    @test run(sess, x)≈1.0
end

#=
@testset "Ipopt" begin
######################### integration with Ipopt.jl #########################
# https://github.com/jainachin/tf-ipopt/blob/master/examples/hs071.py
IPOPT = CustomOptimizer() do f, df, c, dc, x0, x_L, x_U
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
    addOption(prob, "max_iter", 100)

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
minimize(opt, sess)
=#

@testset "newton_raphson_with_grad" begin 
    function f(θ, x)
        x^3 - θ, 3spdiag(x^2)
    end

    θ = [2. .^3;3. ^3; 4. ^3]
    x = newton_raphson_with_grad(f, constant(ones(3)), θ)
    @test run(sess, x)≈[2.;3.;4.]

    θ = constant([2. .^3;3. ^3; 4. ^3])
    x = newton_raphson_with_grad(f, constant(ones(3)), θ)
    @test run(sess, x)≈[2.;3.;4.]

    
    @test run(sess, gradients(sum(x), θ))≈1/3*[2. .^3;3. ^3; 4. ^3] .^(-2/3)
end


# @testset "Constrained Optimizer" begin 
#     reset_default_graph() # this is very important. UnconstrainedOptimizer only works with a fresh session 
#     x = Variable(2*ones(10))
#     y = constant(ones(10))
#     loss = sum((y-x)^4)
#     init(sess)
#     uo = UnconstrainedOptimizer(sess, loss)

#     @test getInit(uo) ≈ 2ones(10)
#     @test getLoss(uo, 3*ones(10))≈ 160.0
#     @test getLossAndGrad(uo, 3*ones(10))[2] ≈  [32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0]
#     @test getLossAndGrad(uo, 3*ones(10))[1] ≈  160.0
# end

# @testset "optimzers" begin
#     ad = AndersonAcceleration()
#     @test_nowarn begin
#         apply!(ad, rand(10), rand(10))
#     end
# end
