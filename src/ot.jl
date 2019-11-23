export sinkhorn, dist, empirical_sinkhorn
# roadmap: implement all related functions from https://github.com/rflamary/POT/blob/master/ot/bregman.py

"""
    sinkhorn(a::Union{PyObject, Array{Float64}}, b::Union{PyObject, Array{Float64}}, M::Union{PyObject, Array{Float64}};
reg::Float64 = 1.0, iter::Int64 = 1000, tol::Float64 = 1e-9, method::String="sinkhorn")

Computes the optimal transport with Sinkhorn algorithm. 
The implementation are adapted from https://github.com/rflamary/POT.  
"""
function sinkhorn(a::Union{PyObject, Array{Float64}}, b::Union{PyObject, Array{Float64}}, M::Union{PyObject, Array{Float64}};
    reg::Float64 = 1.0, iter::Int64 = 1000, tol::Float64 = 1e-9, method::String="sinkhorn")
    if isa(a, Array)
        @assert sum(a)≈1.0
        @assert all(a .>= 0)
    end
    if isa(b, Array)
        @assert sum(b)≈1.0
        @assert all(b .>= 0)
    end
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    M = convert_to_tensor(M)
    reg = convert_to_tensor(reg)
    iter = convert_to_tensor(iter)
    tol = convert_to_tensor(tol)
    sk = load_system_op(COLIB["sinkhorn_knopp"]...; multiple=true)
    if method=="sinkhorn"
        return sk(a,b,M,reg,iter,tol, constant(0))[2]
    else
    end
end

"""
    empirical_sinkhorn(x::Union{PyObject, Array{Float64}}, y::Union{PyObject, Array{Float64}}, dist::Function;
reg::Float64 = 1.0, iter::Int64 = 1000, tol::Float64 = 1e-9, method::String="sinkhorn")

Computes the empirical Wasserstein distance with sinkhorn algorithm. 
The implementation are adapted from https://github.com/rflamary/POT.  
"""
function empirical_sinkhorn(x::Union{PyObject, Array{Float64}}, y::Union{PyObject, Array{Float64}};
    reg::Float64 = 1.0, iter::Int64 = 1000, tol::Float64 = 1e-9, method::String="sinkhorn", dist::Function=dist)
    M = dist(x, y)
    a = tf.ones(tf.shape(x)[1], dtype=tf.float64)/cast(Float64, tf.shape(x)[1])
    b = tf.ones(tf.shape(y)[1], dtype=tf.float64)/cast(Float64, tf.shape(y)[1])
    sinkhorn(a, b, M; reg=reg, iter=iter, tol=tol, method=method)
end


"""
    dist(x::Union{PyObject, Array{Float64}}, y::Union{PyObject, Array{Float64}}, order::Union{Int64, PyObject}=2)

Computes the distance function with norm `order`. `dist` returns a ``n\\times m`` matrix, where ``x\\in \\mathbb{R}^{n\\times d}`` and
``y\\in \\mathbb{R}^{m\\times d}``, and the return ``M\\in \\mathbb{R}^{n\\times m}``
```math
M_{ij} = ||x_i - y_j||_{o}
```
"""
function dist(x::Union{PyObject, Array{Float64}}, y::Union{PyObject, Array{Float64}}, order::Union{Int64, PyObject}=2)
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    order = convert_to_tensor(order)
    ss = load_system_op(COLIB["dist"]...)
    ss(x, y, order)
end