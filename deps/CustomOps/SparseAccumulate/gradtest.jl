using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using Test

# Random.seed!(233)
import ADCME: SparseAssembler

sparse_accumulator = load_op_and_grad("./build/libSparseAccumulator","sparse_accumulator", multiple=false)
sparse_accumulator_add = load_op_and_grad("./build/libSparseAccumulator","sparse_accumulator_add", multiple=false)
sparse_accumulator_copy = load_op_and_grad("./build/libSparseAccumulator","sparse_accumulator_copy", multiple=true)


"""
    SparseAssembler(handle::Union{PyObject, <:Integer}, n::Union{PyObject, <:Integer}, tol::Union{PyObject, <:Real}=0.0)

Creates a SparseAssembler for accumulating `row`, `col`, `val` for sparse matrices. 
- `handle`: an integer handle for creating a sparse matrix. If the handle already exists, `SparseAssembler` return the existing sparse matrix handle. If you are creating different sparse matrices, the handles should be different. 
- `n`: Number of rows of the sparse matrix. 
- `tol` (optional): Tolerance. `SparseAssembler` will treats any values less than `tol` as zero. 

# Example
```julia
handle = SparseAssembler(100, 5, 1e-8)
op1 = accumulate(handle, 1, [1;2;3], [1.0;2.0;3.0])
op2 = accumulate(handle, 2, [1;2;3], [1.0;2.0;3.0])
J = assemble(5, 5, [op1;op2])
```
`J` will be a [`SparseTensor`](@ref) object. 
"""
function SparseAssembler(handle::Union{PyObject, <:Integer}, n::Union{PyObject, <:Integer}, tol::Union{PyObject, <:Real}=0.0)
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
    row = convert_to_tensor(row, dtype=Int32)
    cols = convert_to_tensor(cols, dtype=Int32)
    vals = convert_to_tensor(vals, dtype=Float64)
    return sparse_accumulator_add(acc, row, cols, vals)
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
    if length(size(ops))==0
        ops = reshape(ops, 1)
    end
    ii, jj, vv = sparse_accumulator_copy(ops)
    return SparseTensor(ii, jj, vv, m, n)
end


handle = SparseAssembler(100, 5, 1e-8)
op1 = accumulate(handle, 1, [1;2;3], [1.0;2.0;3.0])
op2 = accumulate(handle, 2, [1;2;3], [1.0;2.0;3.0])
J = assemble(5, 5, [op1;op2])
# J = assemble(acc, 5, 5)
sess = Session(); init(sess)
run(sess, J)

m = 20
n = 100
handle = SparseAssembler(100, m, 0.0)
op = PyObject[]
A = zeros(m, n)
for i = 1:1
    ncol = rand(1:n, 10)
    row = rand(1:m)
    v = rand(10)
    for (k,val) in enumerate(v)
        @show k
        A[row, ncol[k]] += val
    end
    @show v
    push!(op, accumulate(handle, row, ncol, v))
end
op = vcat(op...)
J = assemble(m, n, op)
B = run(sess, J)
@test norm(A-B)<1e-8



handle = SparseAssembler(100, 5, 1.0)
op1 = accumulate(handle, 1, [1;2;3], [2.0;0.5;0.5])
J = assemble(5, 5, op1)
B = run(sess, J)
@test norm(B-[2.0  0.0  0.0  0.0  0.0
    0.0  0.0  0.0  0.0  0.0
    0.0  0.0  0.0  0.0  0.0
    0.0  0.0  0.0  0.0  0.0
    0.0  0.0  0.0  0.0  0.0])<1e-8

handle = SparseAssembler(100, 5, 0.0)
op1 = accumulate(handle, 1, [1;1], [1.0;1.0])
op2 = accumulate(handle, 1, [1;2], [1.0;1.0])

J = assemble(5, 5, [op1;op2])
B = run(sess, J)
@test norm(B-[3.0  1.0  0.0  0.0  0.0
    0.0  0.0  0.0  0.0  0.0
    0.0  0.0  0.0  0.0  0.0
    0.0  0.0  0.0  0.0  0.0
    0.0  0.0  0.0  0.0  0.0])<1e-8

    