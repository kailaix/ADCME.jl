export pde_poisson

@doc raw"""
poisson(node::Union{PyObject, Array{Float64,2}},elem::Union{PyObject, Array{<:Integer,2}},
            bdnode::Union{PyObject, Array{<:Integer}},f::Union{PyObject, Array{Float64}},a::Union{PyObject, Array{Float64}})

Solving the Poisson equation 
```math
-\nabla (a(x) \nabla u(x)) = f(x), \ x \in \Omega \quad u(x) = 0, \ x\in \partial \Omega
```
using finite element method. 

- `node`: $n\times 2$ coordinates array;
- `elem`: $m\times 3$ element index array; 
- `bdnode`: boundary node ids;
- `f`, `a`: $m$ array, values at the element center.
"""
function pde_poisson(node::Union{PyObject, Array{Float64,2}},elem::Union{PyObject, Array{<:Integer,2}},
            bdnode::Union{PyObject, Array{<:Integer}},f::Union{PyObject, Array{Float64}},a::Union{PyObject, Array{Float64}})
        if !haskey(COLIB, "laplace")
            install("Laplace2D")
        end
        ll = load_system_op(COLIB["laplace"]...)
        node,elem,bdnode,f,a = convert_to_tensor([node,elem,bdnode,f,a], [Float64,Int32,Int32,Float64,Float64])
        ll(node,elem,bdnode,f,a)
end