export sinkhorn, dist, empirical_sinkhorn, dtw
# roadmap: implement all related functions from https://github.com/rflamary/POT/blob/master/ot/bregman.py

"""
    sinkhorn(a::Union{PyObject, Array{Float64}}, b::Union{PyObject, Array{Float64}}, M::Union{PyObject, Array{Float64}};
    reg::Float64 = 1.0, iter::Int64 = 1000, tol::Float64 = 1e-9, method::String="sinkhorn")

Computes the optimal transport with Sinkhorn algorithm. 
The implementation are adapted from https://github.com/rflamary/POT.  
"""
function sinkhorn(a::Union{PyObject, Array{Float64}}, b::Union{PyObject, Array{Float64}}, M::Union{PyObject, Array{Float64}};
    reg::Union{PyObject,Float64} = 1.0, iter::Int64 = 1000, tol::Float64 = 1e-9, method::String="sinkhorn", return_optimal::Bool=false)
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
    if method=="sinkhorn"
        sk = load_system_op("sinkhorn_knopp"; multiple=true)
        if return_optimal
            return sk(a,b,M,reg,iter,tol, constant(0))
        end
        return sk(a,b,M,reg,iter,tol, constant(0))[2]
    elseif method=="lp"
        pth = install("OTNetwork")
        lp = load_op_and_grad(pth, "ot_network"; multiple=true)
        if return_optimal
            return lp(a, b, M, iter)
        end
        return lp(a, b, M, iter)[2]
    else
        error("$method not implemented")
    end
end

"""
    empirical_sinkhorn(x::Union{PyObject, Array{Float64}}, y::Union{PyObject, Array{Float64}}, dist::Function;
    reg::Union{PyObject,Float64} = 1.0, iter::Int64 = 1000, tol::Float64 = 1e-9, method::String="sinkhorn")

Computes the empirical Wasserstein distance with sinkhorn algorithm. 
The implementation are adapted from https://github.com/rflamary/POT.  
"""
function empirical_sinkhorn(x::Union{PyObject, Array{Float64}}, y::Union{PyObject, Array{Float64}};
    reg::Union{PyObject,Float64} = 1.0, iter::Int64 = 1000, tol::Float64 = 1e-9, method::String="sinkhorn", dist::Function=dist, return_optimal::Bool=false)
    x, y = convert_to_tensor([x,y], [Float64, Float64])
    length(size(x))==1 && (x = reshape(x, :, 1))
    length(size(y))==1 && (y = reshape(y, :, 1))
    M = dist(x, y)
    a = tf.ones(tf.shape(x)[1], dtype=tf.float64)/cast(Float64, tf.shape(x)[1])
    b = tf.ones(tf.shape(y)[1], dtype=tf.float64)/cast(Float64, tf.shape(y)[1])
    sinkhorn(a, b, M; reg=reg, iter=iter, tol=tol, method=method, return_optimal=return_optimal)
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
    order = convert_to_tensor(order, dtype=Float64)
    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(y, axis=0)
    sum(abs(x-y)^order, dims=3)^(1.0/order)
end

"""
    dtw(s::Union{PyObject, Array{Float64}}, t::Union{PyObject, Array{Float64}}, 
        use_fast::Bool = false)

Computes the dynamic time wrapping (DTW) distance between two time series `s` and `t`. 
Returns the distance and path. `use_fast` specifies whether fast algorithm is used. Note 
fast algorithm may not be accurate.
"""
function dtw(s::Union{PyObject, Array{Float64}}, t::Union{PyObject, Array{Float64}}, 
            use_fast::Bool = false)
    use_fast = Int32(use_fast)
    pth = install("FastDTW")
    dtw_ = load_op_and_grad(pth, "dtw"; multiple=true)
    s,t, use_fast = convert_to_tensor([s,t,use_fast], [Float64,Float64, Int32])
    cost, path = dtw_(s,t,use_fast)
    return cost, path + 1
end