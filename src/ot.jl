export sinkhorn, ot_dist, empirical_sinkhorn, dtw, emd, ot_plot1D, empirical_emd
# roadmap: implement all related functions from https://github.com/rflamary/POT/blob/master/ot/bregman.py

@doc raw"""
    sinkhorn(a::Union{PyObject, Array{Float64}}, b::Union{PyObject, Array{Float64}}, M::Union{PyObject, Array{Float64}};
    reg::Float64 = 1.0, iter::Int64 = 1000, tol::Float64 = 1e-9, method::String="sinkhorn")

Computes the optimal transport with Sinkhorn algorithm. The mathematical formulation is 

```math
\begin{aligned}
\arg\min_P &\ \left(P, M\right) + \lambda \Omega(\Gamma)\\ 
\text{s.t.} &\ \Gamma 1 = a\\ 
&\ \Gamma^T 1 = b\\ 
& \Gamma \geq 0 
\end{aligned}
```
Here $\Omega$ is the entropic regularization. Note if $\lambda$ is very small, the algorithm may encounter numerical instabilities. 

The implementation are adapted from https://github.com/rflamary/POT.  
"""
function sinkhorn(a::Union{PyObject, Array{Float64}}, b::Union{PyObject, Array{Float64}}, M::Union{PyObject, Array{Float64, 2}};
    reg::Union{PyObject,Float64} = 1.0, iter::Int64 = 1000, tol::Float64 = 1e-9, method::String="sinkhorn", returnall::Bool=false)
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
    if method=="lp"
        pth = install("OTNetwork")
        lp = load_op_and_grad(pth, "ot_network"; multiple=true)
        if returnall
            P, loss = lp(a, b, M, iter)
            P = set_shape(P, (length(a), length(b)))
            P, loss 
        end
        return lp(a, b, M, iter)[2]
    else
        METHODIDS = Dict(
            "sinkhorn"=>0, 
            "greenkhorn"=>1
        )
        
        if !haskey(METHODIDS, method)
            error("$method not implemented")
        end 
        METHODID = METHODIDS[method]

        sk = load_system_op("sinkhorn_knopp"; multiple=true)
        if returnall
            return sk(a,b,M,reg,iter,tol, constant(METHODID))
        end
        return sk(a,b,M,reg,iter,tol, constant(METHODID))[2]
    end
end


@doc raw"""
    emd(a::Union{PyObject, Array{Float64}}, b::Union{PyObject, Array{Float64}}, M::Union{PyObject, Array{Float64}};
    iter::Int64 = 1000, tol::Float64 = 1e-9, returnall::Bool=false)

Computes the Earth Mover's Distance, which is defined as 

$$D(M) = \sum_{i=1}^m \sum_{j=1}^n M_{ij} d_{ij}$$

Here $M \in \mathbb{R}^{m\times n}$ is the ground distance matrix. The algorithm solves the following optimization problem 

$$\begin{aligned}\min_{M} &\ D(M)\\\text{s.t.} & \ \sum_{i=1}^m M_{ij} = b_j\\ &\ \sum_{j=1}^n M_{ij} = a_i \end{aligned}$$

The internal solver for the optimization problem is a netflow solver. The algorithm requires $\sum_i a_i = \sum_j b_j = 1$. 
"""
function emd(a::Union{PyObject, Array{Float64}}, b::Union{PyObject, Array{Float64}}, M::Union{PyObject, Array{Float64}};
    iter::Int64 = 1000, tol::Float64 = 1e-9, returnall::Bool=false)
    return sinkhorn(a, b, M, reg = 0.0, iter = iter, tol = tol, method = "lp",
            returnall = returnall)
end


"""
    ot_plot1D(a::Array{Float64, 1}, b::Array{Float64, 1}, M::Array{Float64, 2})

Plots the optimal transport matrix for 1D distributions. 
"""
function ot_plot1D(a::Array{Float64, 1}, b::Array{Float64, 1}, M::Array{Float64, 2})
    @assert length(a) == size(M, 1)
    @assert length(b) == size(M, 2)
    pl = require_import(:PyPlot)
    pl.figure(4, figsize = (5, 5))

    xa = 0:size(M, 1)-1
    xb = 0:size(M, 2)-1

    ax1 = pl.subplot(222)
    pl.plot(xb, b, "r")
    pl.yticks(())
    pl.title("Target")

    ax2 = pl.subplot(223)
    pl.plot(a, xa, "b")
    pl.gca().invert_xaxis()
    pl.gca().invert_yaxis()
    pl.xticks(())
    pl.title("Source")

    pl.subplot(224, sharex=ax1, sharey=ax2)
    pl.imshow(M, interpolation="nearest")
    pl.axis("off")

    pl.xlim((0, size(M, 2)-1))
    pl.tight_layout()
    pl.subplots_adjust(wspace=0., hspace=0.2)

end

@doc raw"""
    empirical_sinkhorn(x::Union{PyObject, Array{Float64}}, y::Union{PyObject, Array{Float64}};
        reg::Union{PyObject,Float64} = 1.0, iter::Int64 = 1000, tol::Float64 = 1e-9, method::String="sinkhorn", dist::Function=dist, returnall::Bool=false)

Computes the empirical Sinkhorn distance with sinkhorn algorithm. Here $x$ and $y$ are samples from two distributions.  

- `reg` (default = 1.0): entropy regularization parameter 
- `tol` (default = 1e-9), `iter` (default = 1000): tolerance and max iterations for the Sinkhorn algorithm 
- `dist` (default = 2): Integer or Function, the distance function between two samples; if `dist` is integer, $L-dist$ norm is used. 
- `returnall`: returns (`TransportMatrix`, `Loss`) if true; otherwise, only `Loss` is returned. 

The implementation are adapted from https://github.com/rflamary/POT.  
"""
function empirical_sinkhorn(x::Union{PyObject, Array{Float64}}, y::Union{PyObject, Array{Float64}};
    reg::Union{PyObject,Float64} = 1.0, iter::Int64 = 1000, tol::Float64 = 1e-9, 
    method::String="sinkhorn", dist::Union{Integer,Function}=2, returnall::Bool=false, normalized::Bool = false)
    x, y = convert_to_tensor([x,y], [Float64, Float64])
    length(size(x))==1 && (x = reshape(x, :, 1))
    length(size(y))==1 && (y = reshape(y, :, 1))
    if isa(dist, Integer)
        d = copy(dist)
        dist = (x,y) -> ot_dist(x, y, d)
    end
    M = dist(x, y)
    if normalized 
        M = M/maximum(M)
    end
    a = tf.ones(tf.shape(x)[1], dtype=tf.float64)/cast(Float64, tf.shape(x)[1])
    b = tf.ones(tf.shape(y)[1], dtype=tf.float64)/cast(Float64, tf.shape(y)[1])
    sinkhorn(a, b, M; reg=reg, iter=iter, tol=tol, method=method, returnall=returnall)
end

"""
    empirical_emd(x::Union{PyObject, Array{Float64}}, y::Union{PyObject, Array{Float64}};
        iter::Int64 = 1000, tol::Float64 = 1e-9, dist::Union{Integer,Function}=2, returnall::Bool=false)

Same as [`empirical_sinkhorn`](@ref), except that the Earth Mover Distance is computed. 
"""
function empirical_emd(x::Union{PyObject, Array{Float64}}, y::Union{PyObject, Array{Float64}};
     iter::Int64 = 1000, tol::Float64 = 1e-9, dist::Union{Integer,Function}=2, returnall::Bool=false, normalized::Bool = false)
     empirical_sinkhorn(x, y; iter = iter, tol = tol, method = "lp",  returnall = returnall)
end


"""
    ot_dist(x::Union{PyObject, Array{Float64}}, y::Union{PyObject, Array{Float64}}, order::Union{Int64, PyObject}=2)

Computes the distance function with norm `order`. `dist` returns a ``n\\times m`` matrix, where ``x\\in \\mathbb{R}^{n\\times d}`` and
``y\\in \\mathbb{R}^{m\\times d}``, and the return ``M\\in \\mathbb{R}^{n\\times m}``
```math
M_{ij} = ||x_i - y_j||_{o}
```
"""
function ot_dist(x::Union{PyObject, Array{Float64}}, y::Union{PyObject, Array{Float64}}, order::Union{Int64, PyObject}=2)
    if !(isa(x, PyObject) || isa(y, PyObject) || isa(order, PyObject))
        x = reshape(x, size(x, 1), :)
        y = reshape(y, size(y, 1), :)
        M = zeros(size(x, 1), size(y, 1))
        for i = 1:size(x, 1)
            for j = 1:size(y, 1)
                M[i, j] = norm(x[i,:]-y[j,:], order)
            end
        end
        return M
    end
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