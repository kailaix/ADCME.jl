using SparseArrays
import Base: accumulate
export SparseTensor, SparseAssembler, spdiag, find, spzero, dense_to_sparse, accumulate, assemble

mutable struct SparseTensor
    o::PyObject
    _diag::Bool
    function SparseTensor(o::PyObject, _diag::Union{Missing,Bool}=missing)
        new(o, _diag)
    end
end

"""
    SparseTensor(I::Union{PyObject,Array{T,1}}, J::Union{PyObject,Array{T,1}}, V::Union{Array{Float64,1}, PyObject}, m::Union{S, PyObject, Nothing}=nothing, n::Union{S, PyObject, Nothing}=nothing) where {T<:Integer, S<:Integer}

Constructs a sparse tensor. 
Examples:
```
ii = [1;2;3;4]
jj = [1;2;3;4]
vv = [1.0;1.0;1.0;1.0]
s = SparseTensor(ii, jj, vv, 4, 4)
s = SparseTensor(sprand(10,10,0.3))
```
"""
function SparseTensor(I::Union{PyObject,Array{T,1}}, J::Union{PyObject,Array{T,1}}, 
      V::Union{Array{Float64,1}, PyObject},
     m::Union{S, PyObject, Nothing}=nothing, n::Union{S, PyObject, Nothing}=nothing; is_diag::Bool=false) where {T<:Integer, S<:Integer}
    if isa(I, PyObject) && size(I,2)==2
        return SparseTensor_(I, J, V)
    end
    I, J, V = convert_to_tensor(I, dtype=Int64), convert_to_tensor(J, dtype=Int64), convert_to_tensor(V)
    m, n = convert_to_tensor(m, dtype=Int64), convert_to_tensor(n, dtype=Int64)
    indices = [I J] .- 1
    value = V
    shape = [m;n]
    sp = tf.SparseTensor(indices, value, shape)
    SparseTensor(tf.sparse.reorder(sp), is_diag)
end

function dense_to_sparse(o::Union{Array, PyObject})
    if isa(o, Array)
        return SparseTensor(sparse(o))
    else
        idx = tf.where(tf.not_equal(o, 0))
        return SparseTensor(tf.SparseTensor(idx, tf.gather_nd(o, idx), o.get_shape()), false)
    end
end

"""
    find(s::SparseTensor)

Returns the row, column and values for sparse tensor `s`.
"""
function find(s::SparseTensor)
    ind = s.o.indices
    val = s.o.values
    ii = ind'[1,:]
    jj = ind'[2,:]
    ii+1, jj+1, val
end


function Base.:copy(s::SparseTensor)
    t = SparseTensor(tf.SparseTensor(copy(s.o.indices), copy(s.o.values), s.o.dense_shape), copy(s.is_diag))
end

function Base.:eltype(o::SparseTensor)
    return (eltype(o.o.indices), eltype(o.o.values))
end

function SparseTensor_(indices::Union{PyObject,Array{T,2}}, value::Union{PyObject,Array{Float64,1}},
        shape::Union{PyObject,Array{T,1}}; is_diag::Bool=false) where T<:Integer
    indices = convert_to_tensor(indices)
    value = convert_to_tensor(value)
    shape = convert_to_tensor(shape)
    sp = tf.SparseTensor(indices-1, value, shape)
    SparseTensor(tf.sparse.reorder(sp), is_diag)
end

"""
    SparseTensor(A::SparseMatrixCSC)
"""
function SparseTensor(A::SparseMatrixCSC)
    rows = rowvals(A)
    vals = nonzeros(A)
    cols = zeros(eltype(rows), length(rows))
    m, n = size(A)
    k = 1
    for i = 1:n
        for j in nzrange(A, i)
            cols[k] = i 
            k += 1
        end
    end
    SparseTensor(rows, cols, vals, m, n; is_diag=isdiag(A))
end

function Base.:show(io::IO, s::SparseTensor)
    shape = size(s)
    s1 = shape[1]===nothing ? "?" : shape[1]
    s2 = shape[2]===nothing ? "?" : shape[2]
    print(io, "SparseTensor($s1, $s2)")
end

function Base.:run(o::PyObject, S::SparseTensor, args...; kwargs...)
    indices, value, shape = run(o, S.o, args...; kwargs...)
    sparse(indices[:,1].+1, indices[:,2].+1, value, shape...)
end

function Base.:Array(S::SparseTensor, args...;kwargs...)
    tf.sparse.to_dense(S.o)
end

function Base.:size(s::SparseTensor)
    (s.o.shape[1].value,s.o.shape[2].value)
end

function Base.:size(s::SparseTensor, i::T) where T<:Integer
    s.o.shape[i].value
end

function PyCall.:+(s::SparseTensor, o::PyObject)
    if size(s)!=size(o)
        error("size $(size(s)) and $(size(o)) does not match")
    end
    out = tf.sparse_add(s.o, o)
    out
end
PyCall.:+(o::PyObject, s::SparseTensor) = s+o
function Base.:-(s::SparseTensor)
    SparseTensor(s.o.indices+1, -s.o.values, s.o.dense_shape, is_diag=s._diag)
end
PyCall.:-(o::PyObject, s::SparseTensor) = o + (-s)
PyCall.:-(s::SparseTensor, o::PyObject) = s + (-o)

Base.:+(s1::SparseTensor, s2::SparseTensor) = SparseTensor(tf.sparse.add(s1.o,s2.o), s1._diag&&s2._diag)
Base.:-(s1::SparseTensor, s2::SparseTensor) = s1 + (-s2)

function Base.:adjoint(s::SparseTensor) 
    indices = [s.o.indices'[2,:] s.o.indices'[1,:]]
    sp = tf.SparseTensor(indices, s.o.values, (size(s,2), size(s,1)))
    sp = tf.sparse.reorder(sp)
    SparseTensor(sp, s._diag)
end

function PyCall.:*(s::SparseTensor, o::PyObject)
    flag = false
    if length(size(o))==1
        flag = true
        o = reshape(o, length(o), 1)
    end
    out = tf.sparse.sparse_dense_matmul(s.o, o)
    if flag
        out = squeeze(out)
    end
    out
end

function Base.:*(s::SparseTensor, o::Array{Float64})
    s*convert_to_tensor(o)
end

function PyCall.:*(o::PyObject, s::SparseTensor)
    if length(size(o))==0
        SparseTensor(tf.SparseTensor(copy(s.o.indices), o*copy(s.o.values), s.o.dense_shape), s._diag)
    else
        tf.sparse.sparse_dense_matmul(s.o, o, adjoint_a=true, adjoint_b=true)'
    end
end

function Base.:*(o::Array{Float64}, s::SparseTensor)
    convert_to_tensor(o)*s
end

function Base.:*(o::Real, s::SparseTensor)
    o = Float64(o)
    SparseTensor(tf.SparseTensor(copy(s.o.indices), o*copy(s.o.values), s.o.dense_shape), s._diag)
end

Base.:*(s::SparseTensor, o::Real) = o*s

Base.:vcat(args::SparseTensor...) = SparseTensor(tf.sparse.concat(0,[s.o for s in args]), false)
Base.:hcat(args::SparseTensor...) = SparseTensor(tf.sparse.concat(1,[s.o for s in args]), false)

function Base.:lastindex(o::SparseTensor, i::Int64)
    return size(o,i)
end

function Base.:getindex(s::SparseTensor, i1::Union{Integer, Colon, UnitRange{T}, PyObject,Array{S,1}},
    i2::Union{Integer, Colon, UnitRange{T}, PyObject,Array{T,1}}) where {S<:Real,T<:Real}
    squeeze_dims = Int64[]
    if isa(i1, Integer); i1 = [i1]; push!(squeeze_dims, 1); end
    if isa(i2, Integer); i2 = [i2]; push!(squeeze_dims, 2); end
    if isa(i1, UnitRange) || isa(i1, StepRange); i1 = collect(i1); end
    if isa(i2, UnitRange) || isa(i2, StepRange); i2 = collect(i2); end
    if isa(i1, Colon); i1 = collect(1:lastindex(s,1)); end
    if isa(i2, Colon); i2 = collect(1:lastindex(s,2)); end
    m_, n_ = length(i1), length(i2)
    i1 = convert_to_tensor(i1, dtype=Int64)
    i2 = convert_to_tensor(i2, dtype=Int64)
    ii1, jj1, vv1 = find(s)
    m = tf.convert_to_tensor(s.o.shape[1],dtype=tf.int64)
    n = tf.convert_to_tensor(s.o.shape[2],dtype=tf.int64)
    ss = load_system_op(COLIB["sparse_indexing"]...)
    ii2, jj2, vv2 = ss(ii1,jj1,vv1,m,n,i1,i2)
    ret = SparseTensor(ii2, jj2, vv2, m_, n_)
    if length(squeeze_dims)>0
        ret = squeeze(Array(ret), dims=squeeze_dims)
    end
    ret
end

function Base.:reshape(s::SparseTensor, shape::T...) where T<:Integer
    SparseTensor(tf.sparse.reshape(s, shape), false)
end

function PyCall.:\(s::SparseTensor, o::PyObject)
    local u
    if length(size(o))!=1
        error("input b must be a vector")
    end
    if size(s,1)!=length(o)
        error("nrows(A) and nrows(b) must match")
    end
    if size(s,1)!=size(s,2)
        # least squre 
        ss = load_system_op(COLIB["sparse_least_square"]...)
        ii = s.o.indices'[1,:]+1
        jj = s.o.indices'[2,:]+1
        ii = cast(ii, Int32)
        jj = cast(jj, Int32)
        vv = cast(s.o.values, Float64)
        o = cast(o, Float64)
        # @show ii, jj, vv, o, constant(size(s, 2), dtype=Int32)
        u = ss(ii, jj, vv, o, constant(size(s, 2), dtype=Int32))
    else
        ss = load_system_op(COLIB["sparse_solver"]...)
        # in case `indices` has dynamical shape
        ii = s.o.indices'[1,:]+1
        jj = s.o.indices'[2,:]+1
        u = ss(ii, jj, s.o.values, constant(collect(1:length(o))),o,
                    constant(size(s, 1)))
    end
    if size(s,2)!=nothing 
        u.set_shape((size(s,2),))
    end
    u
end

Base.:\(s::SparseTensor, o::Array{Float64}) = s\constant(o)

"""
    SparseAssembler(handle::Union{PyObject, <:Integer}, n::Union{PyObject, <:Integer}, tol::Union{PyObject, <:Real}=0.0)

Creates a SparseAssembler for accumulating `row`, `col`, `val` for sparse matrices. 
- `handle`: an integer handle for creating a sparse matrix. If the handle already exists, `SparseAssembler` return the existing sparse matrix handle. If you are creating different sparse matrices, the handles should be different. 
- `n`: Number of rows of the sparse matrix. 
- `tol` (optional): Tolerance. `SparseAssembler` will treats any values less than `tol` as zero. 

# Example 1
```julia
handle = SparseAssembler(100, 5, 1e-8)
op1 = accumulate(handle, 1, [1;2;3], [1.0;2.0;3.0])
op2 = accumulate(handle, 2, [1;2;3], [1.0;2.0;3.0])
J = assemble(5, 5, [op1;op2])
```
`J` will be a [`SparseTensor`](@ref) object. 

# Example 2
```julia
handle = SparseAssembler(0, 5)
op1 = accumulate(handle, 1, [1;2;3], ones(3))
op2 = accumulate(handle, 1, [3], [1.])
op3 = accumulate(handle, 2, [1;3], ones(2))
J = assemble(5, 5, [op1;op2;op3]) # op1, op2, op3 are parallel
Array(run(sess, J))≈[1.0  1.0  2.0  0.0  0.0
                1.0  0.0  1.0  0.0  0.0
                0.0  0.0  0.0  0.0  0.0
                0.0  0.0  0.0  0.0  0.0
                0.0  0.0  0.0  0.0  0.0]
```
"""
function SparseAssembler(handle::Union{PyObject, <:Integer}, n::Union{PyObject, <:Integer}, tol::Union{PyObject, <:Real}=0.0)
    s = load_system_op(COLIB["sparse_assembler"]...; return_str=true)
    sparse_accumulator = load_op(s, "sparse_accumulator")
    n = convert_to_tensor(n, dtype=Int32)
    tol = convert_to_tensor(tol, dtype=Float64)
    handle = convert_to_tensor(handle, dtype=Int32)
    sparse_accumulator(tol, n, handle)
end


"""
    accumulate(handle::PyObject, row::Union{PyObject, <:Integer}, cols::Union{PyObject, Array{<:Integer}}, vals::Union{PyObject, Array{<:Real}})

Accumulates `row`-th row. It adds the value to the sparse matrix
```julia
for k = 1:length(cols)
    A[row, cols[k]] += vals[k]
end
```
`handle` is the handle created by [`SparseAssembler`](@ref). 

See [`SparseAssembler`](@ref) for an example.

!!! Note
    `accumulate` returns a `op::PyObject`. Only when `op` is executed, the nonzero values are populated into the sparse matrix. 
"""
function accumulate(handle::PyObject, row::Union{PyObject, <:Integer}, cols::Union{PyObject, Array{<:Integer}}, 
    vals::Union{PyObject, Array{<:Real}})
    s = load_system_op(COLIB["sparse_assembler"]...; return_str=true)
    sparse_accumulator_add = load_op(s, "sparse_accumulator_add")
    row = convert_to_tensor(row, dtype=Int32)
    cols = convert_to_tensor(cols, dtype=Int32)
    vals = convert_to_tensor(vals, dtype=Float64)
    return sparse_accumulator_add(handle, row, cols, vals)
end

"""
    assemble(m::Union{PyObject, <:Integer}, n::Union{PyObject, <:Integer}, ops::PyObject)

Assembles the sparse matrix from the `ops` created by [`accumulate`](@ref). `ops` is either a single output from `accumulate`, or concated from several `ops`
```julia
op1 = accumulate(handle, 1, [1;2;3], [1.0;2.0;3.0])
op2 = accumulate(handle, 2, [1;2;3], [1.0;2.0;3.0])
op = [op1;op2] # equivalent to `vcat([op1, op2]...)`
```
`m` and `n` are rows and columns of the sparse matrix. 

See [`SparseAssembler`](@ref) for an example.
"""
function assemble(m::Union{PyObject, <:Integer}, n::Union{PyObject, <:Integer}, ops::PyObject)
    s = load_system_op(COLIB["sparse_assembler"]...; return_str=true)
    sparse_accumulator_copy = load_op(s, "sparse_accumulator_copy")
    if length(size(ops))==0
        ops = reshape(ops, 1)
    end
    ii, jj, vv = sparse_accumulator_copy(ops)
    return SparseTensor(ii, jj, vv, m, n)
end




"""
    spdiag(n::Int64)

Constructs a sparse identity matrix of size ``n\\times n``.
"""
function spdiag(n::Int64)
    SparseTensor(sparse(1:n, 1:n, ones(Float64, n)))
end

"""
    spdiag(o::PyObject)

Constructs a sparse diagonal matrix where the diagonal entries are `o`
"""
function spdiag(o::PyObject)
    if length(size(o))!=1
        error("ADCME: input `o` must be a vector")
    end
    ii = collect(1:length(o))
    SparseTensor(ii, ii, o, length(o), length(o), is_diag=true)
end

"""
    spzero(m::Int64, n::Union{Missing, Int64}=missing)

Constructs a empty sparse matrix of size ``m\\times n``. `n=m` if `n` is `missing`
"""
function spzero(m::Int64, n::Union{Missing, Int64}=missing)
    if ismissing(n)
        n = m
    end
    ii = Int64[]
    jj = Int64[]
    vv = Float64[]
    SparseTensor(ii, jj, vv, m, n, is_diag=true)
end


function Base.:*(s1::SparseTensor, s2::SparseTensor)
    ii1, jj1, vv1 = find(s1)
    ii2, jj2, vv2 = find(s2)
    m, n = size(s1)
    n_, k = size(s2)
    if n!=n_
        error("IGACS: matrix size mismatch: ($m, $n) vs ($n_, $k)")
    end
    mat_mul_fn = load_system_op(COLIB["sparse_mat_mul"]...)
    if s1._diag
        mat_mul_fn = load_system_op(COLIB["diag_sparse_mat_mul"]...)
    elseif s2._diag
        mat_mul_fn = load_system_op(COLIB["sparse_diag_mat_mul"]...)
    end
    ii3, jj3, vv3 = mat_mul_fn(ii1-1,jj1-1,vv1,ii2-1,jj2-1,vv2,m,n,k)
    SparseTensor(ii3, jj3, vv3, m, k, is_diag=s1._diag&&s2._diag)
end


# missing is treated as zeros
Base.:+(s1::SparseTensor, s2::Missing) = s1
Base.:-(s1::SparseTensor, s2::Missing) = s1
Base.:*(s1::SparseTensor, s2::Missing) = missing
Base.:/(s1::SparseTensor, s2::Missing) = missing
Base.:-(s1::Missing, s2::SparseTensor) = -s2
Base.:+(s1::Missing, s2::SparseTensor) = s2
Base.:*(s1::Missing, s2::SparseTensor) = missing
Base.:/(s1::Missing, s2::SparseTensor) = missing

function Base.:sum(s::SparseTensor; dims::Union{Integer, Missing}=missing)
    if ismissing(dims)
        tf.sparse.reduce_sum(s.o)
    else
        tf.sparse.reduce_sum(s.o, axis=dims-1)
    end
end