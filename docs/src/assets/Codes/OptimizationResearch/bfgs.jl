using Revise 
using ADCME 
using LinearAlgebra
using LineSearches
using JLD2 

mutable struct BFGSOptimizer <: AbstractOptimizer
    f 
    g!
    x0 
    options
    function BFGSOptimizer()
        new(missing, missing, missing, missing)
    end 
    function BFGSOptimizer(f, g!, x0, options)
        new(f, g!, x0, options)
    end 
end

function ADCME.:optimize(opt::BFGSOptimizer)
    x = opt.x0 
    α0 = 1.0

    losses = Float64[]

    max_num_iter = opt.options[:max_num_iter]
    
    feval, g! = opt.f, opt.g!

    # preallocate memories 
    n = length(x)
    G_ = zeros(n)
    G = zeros(n)
    x_ = zeros(n)
    f_ = 0.0
    f = 0.0
    B = diagm(0=>ones(n))

    # first step: gradient descent 
    g!(G, x)
    f = feval(x)
    
    # the first step is a backtracking linesearch 
    d = -G 
    φ = α->feval(x + α*d)
    dφ = α->begin 
        g = zeros(n)
        g!(g, x + α*d)
        g'*d
    end
    φdφ(x) = φ(x), dφ(x)
    φ0 = φ(0.0)
    dφ0 = dφ(0.0)
    res = BackTracking()(φ, dφ, φdφ, α0, φ0,dφ0)
    α = res[1]


    xnew = x + α * d 
    @. G_ = G
    g!(G, xnew)
    fnew = feval(xnew)
    x, x_ = xnew, x 
    f, f_ = fnew, f 
    push!(losses, f)

    # from second step: BFGS
    for i = 1:max_num_iter-1
        @info "iter $i"
        s = x - x_ 
        y = G - G_ 
        B = (I - s*y'/(y'*s)) * B * (I - y*s'/(y'*s)) + s*s'/(y'*s)
        dx = -B*G 
        
        # line search 
        d = dx
        φ = α->feval(x + α*d)
        dφ = α->begin 
            g = zeros(n)
            g!(g, x + α*d)
            g'*d
        end
        φdφ(x) = φ(x), dφ(x)
        φ0 = φ(0.0)
        dφ0 = dφ(0.0)
        # scaled_α0 = 0.01/maximum(abs.(dx))
        res = BackTracking()(φ, dφ, φdφ, α0, φ0,dφ0)
        α = res[1]


        xnew = x + α * dx
        @. G_ = G
        g!(G, xnew)
        fnew = feval(xnew)
        x, x_ = xnew, x 
        f, f_ = fnew, f
        
        # check for convergence 
        if norm(G)<opt.options[:g_tol] || isnan(f)
            break
        end
        push!(losses, f)
    end

    return losses
end

using Random; Random.seed!(233)

x = rand(10)
y = sin.(x)
θ = Variable(ae_init([1,20,20,20,1]))
z = squeeze(fc(x, [20, 20, 20, 1], θ))

loss = sum((z-y)^2)
sess = Session(); init(sess)
losses = Optimize!(sess, loss; optimizer = BFGSOptimizer(), max_num_iter=2000)

@save "data/bfgs.jld2" losses 


