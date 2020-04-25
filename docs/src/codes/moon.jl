using Random

# `nmoons` is adapted from https://github.com/wildart/nmoons
function nmoons(::Type{T}, n::Int=100, c::Int=2;
                shuffle::Bool=false, ε::Real=0.1, d::Int = 2,
                translation::Vector{T}=zeros(T, d),
                rotations::Dict{Pair{Int,Int},T} = Dict{Pair{Int,Int},T}(),
                seed::Union{Int,Nothing}=nothing) where {T <: Real}
    rng = seed === nothing ? Random.GLOBAL_RNG : MersenneTwister(Int(seed))
    ssize = floor(Int, n/c)
    ssizes = fill(ssize, c)
    ssizes[end] += n - ssize*c
    @assert sum(ssizes) == n "Incorrect partitioning"
    pi = convert(T, π)
    R(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    X = zeros(d,0)
    for (i, s) in enumerate(ssizes)
        circ_x = cos.(range(zero(T), pi, length=s)).-1.0
        circ_y = sin.(range(zero(T), pi, length=s))
        C = R(-(i-1)*(2*pi/c)) * hcat(circ_x, circ_y)'
        C = vcat(C, zeros(d-2, s))
        dir = zeros(d)-C[:,end] # translation direction
        #dir ./= abs(sum(dir))
        @debug "Dimension $i"  direction=dir
        X = hcat(X, C .+ dir.*translation)
    end
    y = vcat([fill(i,s) for (i,s) in enumerate(ssizes)]...)
    if shuffle
        idx = randperm(rng, n)
        X, y = X[:, idx], y[idx]
    end
    # Add noise to the dataset
    if ε > 0.0
        #using Distributions
        #Nz = Normal(zero(T), convert(T,ε/d))
        #X += rand(rng, Nz, size(X))
        X += randn(rng, size(X)).*convert(T,ε/d)
    end
    # Rotate dataset
    for ((i,j),θ) in rotations
        X[[i,j],:] .= R(θ)*view(X,[i,j],:)
    end
    return X, y
end