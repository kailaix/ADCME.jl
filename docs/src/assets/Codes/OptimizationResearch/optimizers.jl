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

    if haskey(opt.options, :B)
        @info "Found Hessian matrix "
        B = opt.options[:B]
    else 
        B = diagm(0=>ones(n))
    end

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
        # α = min(1.0, α)
        α0 = min(10.0, 10α)
        res = BackTracking()(φ, dφ, φdφ, α0, φ0,dφ0)
        α = res[1]
        @info α

        if abs(α)<1e-15
            @warn "Step size too small"
            return losses 
        end


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




mutable struct LBFGSOptimizer <: AbstractOptimizer
    f 
    g!
    x0 
    options
    function LBFGSOptimizer()
        new(missing, missing, missing, missing)
    end 
    function LBFGSOptimizer(f, g!, x0, options)
        new(f, g!, x0, options)
    end 
end

function ADCME.:optimize(opt::LBFGSOptimizer)
    x = opt.x0 
    α0 = 1.0

    losses = Float64[]

    max_num_iter = opt.options[:max_num_iter]
    lbfgs_storage = opt.options[:m]
    
    feval, g! = opt.f, opt.g!

    # preallocate memories 
    Ss = Vector{Float64}[]
    Ys = Vector{Float64}[]
    αs = zeros(lbfgs_storage)

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
    pushfirst!(Ss, x - x_ )
    pushfirst!(Ys, G - G_ )

    # from second step: BFGS
    for i = 1:max_num_iter-1
        @info "iter $i"
        dx = -G 
        for j = 1:length(Ss)
            αs[j] = Ss[j]'*dx/(Ys[j]'*Ss[j])
            dx -= αs[j]*Ys[j]
        end
        dx = Ys[1]'*Ss[1]/(Ys[1]'*Ys[1]) * dx
        for j = length(Ss):-1:1
            β = Ys[j]'*dx / (Ys[j]'*Ss[j])
            dx += (αs[j] - β) * Ss[j]
        end
        
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
        # α = min(1.0, α)
        α0 = min(1.0, 10α)
        res = BackTracking()(φ, dφ, φdφ, α0, φ0,dφ0)
        α = res[1]

        if abs(α)<1e-15
            @warn "Step size too small"
            return losses 
        end


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
        pushfirst!(Ss, x - x_ )
        pushfirst!(Ys, G - G_ )
        if length(Ss)>lbfgs_storage
            pop!(Ss)
            pop!(Ys)
        end
    end

    return losses
end
