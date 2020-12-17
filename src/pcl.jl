# This is an implementation of second order physics constrained learning 

export pcl_sparse_solve, pcl_square_sum, pcl_hessian, pcl_linear_op, pcl_compress

@doc raw"""
    pcl_sparse_solve(indices::Array{Int64, 2}, 
        vals::Array{Float64, 1}, 
        u::Array{Float64, 1}, 
        hessian_u::Array{Float64, 2}, grad_u::Array{Float64, 1})

Hessian backpropagation for 

$$Au = f$$

`A` is given by a sparse matrix (the nonzero entries are a vector $a$). This function 
back-propagates the Hessian w.r.t. $u$ to the Hessian w.r.t. $A$. 
"""
function pcl_sparse_solve(indices::Array{Int64, 2}, 
    vals::Array{Float64, 1}, 
    u::Array{Float64, 1}, 
    hessian_u::Array{Float64, 2}, grad_u::Array{Float64, 1})
    A = sparse(indices[:,1], indices[:,2], vals)
    indof = length(vals)
    outdof = length(u)
    invA = inv(Array(A))
    J = zeros(indof, outdof)
    for k = 1:indof
        i = indices[k,1]
        j = indices[k,2]
        J[k,:] = -invA[:,i] * u[j]
    end
    H = zeros(indof, indof)
    x = A'\grad_u
    for l = 1:indof
        for r = 1:indof
            li, lj = indices[l, :]
            ri, rj = indices[r, :]
            H[l, r] = -(x[li] * J[r, lj] + x[ri] * J[l, rj])
        end
    end
    return J*hessian_u*J' + H
end

@doc raw"""
    pcl_square_sum(scaling::Float64 = 1.0)

Returns the Hessian matrix for 
$$\text{scaling} * \|u - u_o\|_2^2$$

`pcl_square_sum` is a reduction op, which can provides the highest level Hessian matrix. 
"""
pcl_square_sum(n::Int64; scaling::Float64 = 1.0) = 2Array(I(n)) * scaling


@doc raw"""
    pcl_linear_op(J::Array{Float64, 2}, W::Array{Float64,2})

For a linear operator 

$$y = J^Tx + b$$

The PCL backpropagation is given by 
$$H = JWJ^T$$
"""
function pcl_linear_op(J::Array{Float64, 2}, W::Array{Float64,2})
    J * W * J'
end

@doc raw"""
    pcl_hessian(y::PyObject, x::PyObject)

Returns the Hessian tensor for the operator 

$$y = F(x)$$

$x\in \mathbb{R}^m$ and $y\in \mathbb{R}^n$ are 1D vectors. The function returns a tuple 

- `H`: $m\times m$ Hessian matrix
- `W`: $n\times n$ Hessian matrix from downstream of the computational graph 
- `dy`: length $n$ tensor; gradient from downstream of the computational graph
"""
function pcl_hessian(y::PyObject, x::PyObject, loss::PyObject)
    @assert length(size(x))==length(size(y))==1
    J = jacobian(y, x)
    if isnothing(J)
        error("`jacobian(y, x) = nothing. Please check the dependency of `y` on `x`")
    end
    W = placeholder(Float64, shape = (length(y), length(y)))
    dy = independent(gradients(loss, y))
    H = J' * W * J + hessian(dot(dy, y), x)
    return H, W
end

@doc raw"""
    pcl_compress(indices::Array{Int64, 2})

Computes the Jacobian matrix for `compress`. Assume that `compress` does the following transformation:
```
indices, values (v) --> new_indices, new_values (u)
```
This function computes 
$$J_{ij} = \frac{\partial u_j}{\partial v_i}$$
"""
function pcl_compress(indices::Array{Int64, 2})
    N = size(indices,1)
    v = zeros(N)
    nout = zeros(Int32,1)
    indices = indices'[:] .- 1
    out = @eval ccall((:pcl_SparseCompressor, $(ADCME.LIBADCME)), Ptr{Cdouble}, 
        (Ptr{Clonglong}, Ptr{Cdouble}, Cint, Ptr{Cint}),
         $indices, $v, Int32($N), $nout)
    J = unsafe_wrap(Array, out, (N,Int64(nout[1])); own = true)
    return J 
end