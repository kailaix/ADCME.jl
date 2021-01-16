export test_jacobian, test_gradients, linedata, lineview, meshdata, 
    meshview, gradview, jacview, PCLview, pcolormeshview,
    animate, saveanim, test_hessian

function require_pyplot()
    if !isdefined(Main, :PyPlot)
        error("You must load PyPlot to use this function, e.g., `using PyPlot` or `import PyPlot`")
    end
    Main.PyPlot
end


"""
    test_gradients(f::Function, x0::Array{Float64, 1}; scale::Float64 = 1.0, showfig::Bool = true)

Testing the gradients of a vector function `f`:
`y, J = f(x)` where `y` is a scalar output and `J` is the vector gradient.
"""
function test_gradients(f::Function, x0::Array{Float64, 1}; 
    scale::Float64 = 1.0, showfig::Bool = true, mpi::Bool = false)
    
    v0 = rand(Float64,length(x0))
    γs = scale ./10 .^(1:5)
    err2 = Float64[]
    err1 = Float64[]
    f0, J = f(x0)
    for i = 1:5
        f1, _ = f(x0+γs[i]*v0)
        push!(err1, abs(f1-f0))
        push!(err2, abs(f1-f0-γs[i]*sum(J.*v0)))
    end
    if showfig 
        if mpi && mpi_rank()>0
            return 
        end
        mp = require_pyplot()
        mp.close("all")
        mp.loglog(γs, err1, label="Finite Difference")
        mp.loglog(γs, err2, label="Automatic Differentiation")
        mp.loglog(γs, γs * 0.5*abs(err1[1])/γs[1], "--",label="\$\\mathcal{O}(\\gamma)\$")
        mp.loglog(γs, γs.^2 * 0.5*abs(err2[1])/γs[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
        mp.plt.gca().invert_xaxis()
        mp.legend()
        mp.savefig("test.png")
        @info "Results saved to test.png"
        println("Finite difference: $err1")
        println("Automatic differentiation: $err2")
    end
    return err1, err2
end

"""
    test_jacobian(f::Function, x0::Array{Float64, 1}; scale::Float64 = 1.0, showfig::Bool = true)

Testing the gradients of a vector function `f`:
`y, J = f(x)` where `y` is a vector output and `J` is the Jacobian.
"""
function test_jacobian(f::Function, x0::Array{Float64, 1}; scale::Float64 = 1.0, showfig::Bool = true)
    
    v0 = rand(Float64,size(x0))
    γs = scale ./10 .^(1:5)
    err2 = Float64[]
    err1 = Float64[]
    f0, J = f(x0)
    for i = 1:5
        f1, _ = f(x0+γs[i]*v0)
        push!(err1, norm(f1-f0))
        push!(err2, norm(f1-f0-γs[i]*J*v0))
    end
    if showfig
        mp = require_pyplot()
        mp.close("all")
        mp.loglog(γs, err1, label="Finite Difference")
        mp.loglog(γs, err2, label="Automatic Differentiation")
        mp.loglog(γs, γs * 0.5*abs(err1[1])/γs[1], "--",label="\$\\mathcal{O}(\\gamma)\$")
        mp.loglog(γs, γs.^2 * 0.5*abs(err2[1])/γs[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
        mp.plt.gca().invert_xaxis()
        mp.legend()
        mp.savefig("test.png")
        @info "Results saved to test.png"
        println("Finite difference: $err1")
        println("Automatic differentiation: $err2")
    end
    return err1, err2
end


"""
    test_hessian(f::Function, x0::Array{Float64, 1}; scale::Float64 = 1.0)

Testing the Hessian of a scalar function `f`:
`g, H = f(x)` where `y` is a scalar output, `g` is a vector gradient output, and `H` is the Hessian.
"""
test_hessian(f::Function, x0::Array{Float64, 1}; kwargs...) = test_jacobian(f, x0;  kwargs...)


function linedata(θ1, θ2=nothing; n::Integer = 10)
    if θ2 === nothing
        θ2 = θ1 .* (1 .+ randn(size(θ1)...))
    end
    T = []
    for x in LinRange(0.0,1.0,n)
        push!(T, θ1*(1-x)+θ2*x)
    end
    T
end

function lineview(losses::Array{Float64})
    mp = require_pyplot()
    n = length(losses)
    v = collect(LinRange(0.0,1.0,n))
    mp.close("all")
    mp.plot(v, losses)
    mp.xlabel("\$\\alpha\$")
    mp.ylabel("loss")
    mp.grid("on")
end


@doc raw"""
    lineview(sess::PyObject, pl::PyObject, loss::PyObject, θ1, θ2=nothing; n::Integer = 10)

Plots the function 

$$h(α) = f((1-α)θ_1 + αθ_2)$$

# Example
```julia
pl = placeholder(Float64, shape=[2])
l = sum(pl^2-pl*0.1)
sess = Session(); init(sess)
lineview(sess, pl, l, rand(2))
```
"""
function lineview(sess::PyObject, pl::PyObject, loss::PyObject, θ1, θ2=nothing; n::Integer = 10)
    mp = require_pyplot()
    dat = linedata(θ1, θ2, n=n)
    V = zeros(length(dat))
    for i = 1:length(dat)
        @info i, n 
        V[i] = run(sess, loss, pl=>dat[i])
    end
    lineview(V)
end

function meshdata(θ;
     a::Real=1, b::Real=1, m::Integer=10, n::Integer=10 ,direction::Union{Array{<:Real}, Missing}=missing)
    local δ, γ
    as = LinRange(-a, a, m)
    bs = LinRange(-b, b, n)
    α = zeros(m, n)
    β = zeros(m, n)
    θs = Array{Any}(undef, m, n)
    if !ismissing(direction)
        δ = direction - θ
        γ = randn(size(θ)...)
        γ = γ/norm(γ)*norm(δ)
    else
        δ = randn(size(θ)...)
        γ = randn(size(θ)...)
    end
    for i = 1:m 
        for j = 1:n 
            α[i,j] = as[i]
            β[i,j] = bs[j]
            θs[i,j] = θ + δ*as[i] + γ*bs[j]
        end
    end
    return θs
end

function meshview(losses::Array{Float64}, a::Real=1, b::Real=1)
    mp = require_pyplot()
    m, n = size(losses)
    α = zeros(m, n)
    β = zeros(m, n)
    as = LinRange(-a, a, m)
    bs = LinRange(-b, b, n)
    for i = 1:m 
        for j = 1:n 
            α[i,j] = as[i]
            β[i,j] = bs[j]
        end
    end
    mp.close("all")
    mp.mesh(α, β, losses)
    mp.xlabel("\$\\alpha\$")
    mp.ylabel("\$\\beta\$")
    mp.zlabel("loss")
    mp.scatter3D(α[(m+1)÷2, (n+1)÷2], β[(m+1)÷2, (n+1)÷2], losses[(m+1)÷2, (n+1)÷2], color="red", s=100)
    return α, β, losses
end

function pcolormeshview(losses::Array{Float64}, a::Real=1, b::Real=1)
    mp = require_pyplot()
    m, n = size(losses)
    α = zeros(m, n)
    β = zeros(m, n)
    as = LinRange(-a, a, m)
    bs = LinRange(-b, b, n)
    for i = 1:m 
        for j = 1:n 
            α[i,j] = as[i]
            β[i,j] = bs[j]
        end
    end
    mp.close("all")
    mp.pcolormesh(α, β, losses)
    mp.xlabel("\$\\alpha\$")
    mp.ylabel("\$\\beta\$")
    mp.scatter(0.0,0.0, marker="*", s=100)
    mp.colorbar()
    return α, β, losses
end

function meshview(sess::PyObject, pl::PyObject, loss::PyObject, θ; 
        a::Real=1, b::Real=1, m::Integer=9, n::Integer=9, 
        direction::Union{Array{<:Real}, Missing}=missing)
    dat = meshdata(θ; a=a, b=b, m=m, n=n, direction=direction)
    m, n = size(dat)
    V = zeros(m, n)
    for i = 1:m 
        @info i, m
        for j = 1:n 
            V[i,j] = run(sess, loss, pl=>dat[i,j])
        end
    end
    meshview(V, a, b)
end

function pcolormeshview(sess::PyObject, pl::PyObject, loss::PyObject, θ; 
    a::Real=1, b::Real=1, m::Integer=9, n::Integer=9, 
    direction::Union{Array{<:Real}, Missing}=missing)
    dat = meshdata(θ; a=a, b=b, m=m, n=n, direction=direction)
    m, n = size(dat)
    V = zeros(m, n)
    for i = 1:m 
        @info i, m
        for j = 1:n 
            V[i,j] = run(sess, loss, pl=>dat[i,j])
        end
    end
    pcolormeshview(V, a, b)
end


function gradview(sess::PyObject, pl::PyObject, loss::PyObject, u0, grad::PyObject; 
    scale::Float64 = 1.0, mpi::Bool = false)
    mp = require_pyplot()
    local v 
    if length(size(u0))==0
        v = rand()
    else
        v = rand(length(u0))
    end
    γs = scale ./ 10 .^ (1:5)
    v1 = Float64[]
    v2 = Float64[]
    L_ = run(sess, loss, pl=>u0)
    J_ = run(sess, grad, pl=>u0)
    for i = 1:5
        @info i 
        L__ = run(sess, loss, pl=>u0+v*γs[i])
        push!(v1, norm(L__-L_))
        if size(J_)==size(v)
            if length(size(J_))==0
                push!(v2, norm(L__-L_-γs[i]*sum(J_*v)))
            else
                push!(v2, norm(L__-L_-γs[i]*sum(J_.*v)))
            end
        else
            push!(v2, norm(L__-L_-γs[i]*J_*v))
        end

    end
    if !(mpi) || (mpi && mpi_rank()==0)
        mp.close("all")
        mp.loglog(γs, abs.(v1), "*-", label="finite difference")
        mp.loglog(γs, abs.(v2), "+-", label="automatic linearization")
        mp.loglog(γs, γs.^2 * 0.5*abs(v2[1])/γs[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
        mp.loglog(γs, γs * 0.5*abs(v1[1])/γs[1], "--",label="\$\\mathcal{O}(\\gamma)\$")
        mp.plt.gca().invert_xaxis()
        mp.legend()
        mp.xlabel("\$\\gamma\$")
        mp.ylabel("Error")
        if mpi 
            mp.savefig("mpi_gradview.png")
        end
    end

    return v1, v2
end


"""
    gradview(sess::PyObject, pl::PyObject, loss::PyObject, u0; scale::Float64 = 1.0)

Visualizes the automatic differentiation and finite difference convergence converge. For correctly implemented
differentiable codes, the convergence rate for AD should be 2 and for FD should be 1 (if not evaluated at stationary point).

- `scale`: you can control the step size for perturbation. 
"""
function gradview(sess::PyObject, pl::PyObject, loss::PyObject, u0; scale::Float64 = 1.0, mpi::Bool = false)
    grad = tf.convert_to_tensor(gradients(loss, pl))
    gradview(sess, pl, loss, u0, grad, scale = scale, mpi = mpi)
end


@doc raw"""
    jacview(sess::PyObject, f::Function, θ::Union{Array{Float64}, PyObject, Missing}, 
    u0::Array{Float64}, args...)

Performs gradient test for a vector function. `f` has the signature 
```
f(θ, u) -> r, J
```
Here `θ` is a nuisance  parameter, `u` is the state variables (w.r.t. which the Jacobian is computed),
`r` is the residual vector, and `J` is the Jacobian matrix (a dense matrix or a [`SparseTensor`](@ref)).

# Example 1
```julia
u0 = rand(10)
function verify_jacobian_f(θ, u)
    r = u^3+u - u0
    r, spdiag(3u^2+1.0)
end
jacview(sess, verify_jacobian_f, missing, u0)
```

# Example 2
```
u0 = rand(10)
rs = rand(10)
function verify_jacobian_f(θ, u)
    r = [u^2;u] - [rs;rs]
    r, [spdiag(2*u); spdiag(10)]
end
jacview(sess, verify_jacobian_f, missing, u0); close("all")
```
"""
function jacview(sess::PyObject, f::Function, θ::Union{Array{Float64}, PyObject, Missing}, 
                u0::Array{Float64}, args...)
    mp = require_pyplot()
    u = placeholder(Float64, shape=[length(u0)])
    L, J = f(θ, u)
    init(sess)
    L_ = run(sess, L, u=>u0, args...)
    J_ = run(sess, J, u=>u0, args...)
    v = rand(length(u0))
    γs = 1.0 ./ 10 .^ (1:5)
    v1 = Float64[]
    v2 = Float64[]
    for i = 1:5
        L__ = run(sess, L, u=>u0+v*γs[i], args...)
        push!(v1, norm(L__-L_))
        push!(v2, norm(L__-L_-γs[i]*J_*v))
    end
    mp.close("all")
    mp.loglog(γs, abs.(v1), "*-", label="finite difference")
    mp.loglog(γs, abs.(v2), "+-", label="automatic linearization")
    mp.loglog(γs, γs.^2 * 0.5*abs(v2[1])/γs[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
    mp.loglog(γs, γs * 0.5*abs(v1[1])/γs[1], "--",label="\$\\mathcal{O}(\\gamma)\$")
    mp.plt.gca().invert_xaxis()
    mp.legend()
    mp.xlabel("\$\\gamma\$")
    mp.ylabel("Error")
end


function PCLview(sess::PyObject, f::Function, L::Function, θ::Union{PyObject,Array{Float64,1}, Float64}, 
    u0::Union{PyObject, Array{Float64}}, args...; options::Union{Dict{String, T}, Missing}=missing) where T<:Real
    mp = require_pyplot()
    if isa(θ, PyObject)
        θ = run(sess, θ, args...)
    end
    x = placeholder(Float64, shape=[length(θ)])
    l, u, g = NonlinearConstrainedProblem(f, L, x, u0; options=options)
    init(sess)
    L_ = run(sess, l, x=>θ, args...)
    J_ = run(sess, g, x=>θ, args...)
    v = rand(length(x))
    γs = 1.0 ./ 10 .^ (1:5)
    v1 = Float64[]
    v2 = Float64[]
    for i = 1:5
        @info i 
        L__ = run(sess, l, x=>θ+v*γs[i], args...)
        # @show L__,L_,J_, v
        push!(v1, L__-L_)
        push!(v2, L__-L_-γs[i]*sum(J_.*v))
    end
    mp.close("all")
    mp.loglog(γs, abs.(v1), "*-", label="finite difference")
    mp.loglog(γs, abs.(v2), "+-", label="automatic linearization")
    mp.loglog(γs, γs.^2 * 0.5*abs(v2[1])/γs[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
    mp.loglog(γs, γs * 0.5*abs(v1[1])/γs[1], "--",label="\$\\mathcal{O}(\\gamma)\$")
    mp.plt.gca().invert_xaxis()
    mp.legend()
    mp.xlabel("\$\\gamma\$")
    mp.ylabel("Error")
end

@doc raw"""
    animate(update::Function, frames; kwargs...)

Creates an animation using update function `update`. 
# Example 
```julia
θ = LinRange(0, 2π, 100)
x = cos.(θ)
y = sin.(θ)
pl, = plot([], [], "o-")
t = title("0")
xlim(-1.2,1.2)
ylim(-1.2,1.2)
function update(i)
    t.set_text("$i")
    pl.set_data([x[1:i] y[1:i]]'|>Array)
end
animate(update, 1:100)
```
"""
function animate(update::Function, frames; kwargs...)
    mp = require_pyplot()
    animation_ = pyimport("matplotlib.animation")
    if !isa(frames, Integer)
        frames = Array(frames)
    end
    animation_.FuncAnimation(mp.gcf(), update, frames=frames; kwargs...)
end


"""
    saveanim(anim::PyObject, filename::String; kwargs...)

Saves the animation produced by [`animate`](@ref)
"""
function saveanim(anim::PyObject, filename::String; kwargs...)
    if Sys.iswindows()
        anim.save(filename, "ffmpeg"; kwargs...)
    else
        anim.save(filename, "imagemagick"; kwargs...)
    end
end