export RBF2D

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