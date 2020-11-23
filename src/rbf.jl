export RBF2D, interp1

@doc raw"""
    function RBF2D(xc::Union{PyObject, Array{Float64, 1}}, yc::Union{PyObject, Array{Float64, 1}}; 
        c::Union{PyObject, Array{Float64, 1}, Missing} = missing, 
        eps::Union{PyObject, Array{Float64, 1}, Real, Missing} = missing,
        d::Union{PyObject, Array{Float64, 1}} = zeros(0), 
        kind::Int64 = 0)

Constructs a radial basis function representation on a 2D domain

$$f(x, y) = \sum_{i=1}^N c_i \phi(r; \epsilon_i) + d_0 + d_1 x + d_2 y$$

Here `d` can be either 0, 1 (only $d_0$ is present), or 3 ($d_0$, $d_1$, and $d_2$ are all present).

`kind` determines the type of radial basis functions 

* 0:Gaussian

$$\phi(r; \epsilon) = e^{-(\epsilon r)^2}$$

* 1:Multiquadric

$$\phi(r; \epsilon) = \sqrt{1+(\epsilon r)^2}$$

* 2:Inverse quadratic

$$\phi(r; \epsilon) = \frac{1}{1+(\epsilon r)^2}$$

* 3:Inverse multiquadric

$$\phi(r; \epsilon) = \frac{1}{\sqrt{1+(\epsilon r)^2}}$$

Returns a callable struct, i.e. to evaluates the function at locations $(x, y)$ (`x` and `y` are both vectors), run 
```julia
rbf(x, y)
```
"""
mutable struct RBF2D
    xc::PyObject
    yc::PyObject
    eps::PyObject
    c::PyObject
    d::PyObject
    kind::Int64

    function RBF2D(xc::Union{PyObject, Array{Float64, 1}}, yc::Union{PyObject, Array{Float64, 1}}; 
            c::Union{PyObject, Array{Float64, 1}, Missing} = missing, 
            eps::Union{PyObject, Array{Float64, 1}, Real, Missing} = missing,
            d::Union{PyObject, Array{Float64, 1}} = zeros(0), 
            kind::Int64 = 0)
        if isa(eps, Real) 
            eps = eps * ones(length(xc))
        end
        c = coalesce(c, Variable(zeros(length(xc))))
        eps = coalesce(eps, ones(length(xc)))
        @assert length(xc)==length(yc)==length(c)==length(eps)
        @assert length(d) in [0,1,3]
        new(constant(xc), constant(yc), constant(eps), constant(c), constant(d), kind)
    end
end

function (o::RBF2D)(x,y)
    radial_basis_function_ = load_op_and_grad(libadcme,"radial_basis_function")
    x,y,xc,yc,eps,c,d,kind = convert_to_tensor(Any[x,y,o.xc,o.yc,o.eps,o.c,o.d,o.kind], [Float64,Float64,Float64,Float64,Float64,Float64,Float64,Int64])
    out = radial_basis_function_(x,y,xc,yc,eps,c,d,kind)
    set_shape(out, (length(x,)))
end

function Base.:show(io::IO, o::RBF2D)
    ct = hasproperty(o.c, :trainable) && o.c.trainable
    et = hasproperty(o.eps, :trainable) && o.eps.trainable
    dt = hasproperty(o.d, :trainable) && o.d.trainable
    print("RadialBasisFunction(NumOfCenters=$(length(o.xc)),NumOfAdditionalTerm=$(length(o.d)),CoeffIsVariable=$(ct),ShapeIsVariable=$(et),AdditionalTermIsVariable=$dt)")
end

"""
    interp1(x::Union{Array{Float64, 1}, PyObject},v::Union{Array{Float64, 1}, PyObject},xq::Union{Array{Float64, 1}, PyObject})

returns interpolated values of a 1-D function at specific query points using linear interpolation. 
Vector x contains the sample points, and v contains the corresponding values, v(x). 
Vector xq contains the coordinates of the query points.

!!! info 
    `x` should be sorted in ascending order. 

# Example
```julia
x = sort(rand(10))
y = constant(@. x^2 + 1.0)
z = [x[1]; x[2]; rand(5) * (x[end]-x[1]) .+ x[1]; x[end]]
u = interp1(x,y,z)
```
"""
function interp1(x::Union{Array{Float64, 1}, PyObject},v::Union{Array{Float64, 1}, PyObject},xq::Union{Array{Float64, 1}, PyObject})
    interp_dim_one_ = load_system_op("interp_dim_one")
    x,v,xq = convert_to_tensor(Any[x,v,xq], [Float64,Float64,Float64])
    out = interp_dim_one_(x,v,xq)
end