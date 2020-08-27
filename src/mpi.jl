export mpi_bcast, mpi_init, mpi_recv, mpi_send, 
    mpi_sendrecv, mpi_sum,  mpi_finalize, mpi_initialized, mpi_halo_exchange,
    mpi_finalized, mpi_rank, mpi_size, mpi_sync!, mpi_gather, mpi_SparseTensor

"""
    mpi_init()

Initialized the MPI session. `mpi_init` must be called before any `run(sess, ...)`.
"""
function mpi_init()
    if mpi_initialized()
        @warn "MPI has already been initialized"
        return 
    end
    @eval ccall((:mpi_init, $LIBADCME), Cvoid, ())
end

"""
    mpi_finalize()

Finalize the MPI call.
"""
function mpi_finalize()
    mpi_check()
    @eval ccall((:mpi_finalize, $LIBADCME), Cvoid, ())
end

"""
    mpi_rank()

Returns the rank of current MPI process (rank 0 based).
"""
function mpi_rank()
    mpi_check()
    out = @eval ccall((:mpi_rank, $LIBADCME), Cint, ())
    Int64(out)
end

"""
    mpi_size()

Returns the size of MPI world.
"""
function mpi_size()
    mpi_check()
    @eval ccall((:mpi_size, $LIBADCME), Cint, ())
end


"""
    mpi_finalized()

Returns a boolean indicating whether the current MPI session is finalized.
"""
function mpi_finalized()
    if !mpi_initialized()
        error("MPI has not been initialized.")
    end 
    Bool(@eval ccall((:mpi_finalized, $LIBADCME), Cuchar, ()))
end

"""
    mpi_initialized()

Returns a boolean indicating whether the current MPI session is initialized.
"""
function mpi_initialized()
    Bool(@eval ccall((:mpi_initialized, $LIBADCME), Cuchar, ()))
end

"""
    mpi_sync!(message::Array{Int64,1}, root::Int64 = 0)
    mpi_sync!(message::Array{Float64,1}, root::Int64 = 0)

Sync `message` across all MPI processors.
"""
function mpi_sync!(message::Array{Int64,1}, root::Int64 = 0)
    mpi_check()
    @eval ccall((:mpi_sync, $LIBADCME), Cvoid, (Ptr{Clonglong}, Cint, Cint), $message, Int32(length($message)), Int32($root))
end

function mpi_sync!(message::Array{Float64,1}, root::Int64 = 0)
    mpi_check()
    @eval ccall((:mpi_sync_double, $LIBADCME), Cvoid, (Ptr{Cdouble}, Cint, Cint), $message, Int32(length($message)), Int32($root))
end

function mpi_check()
    if !mpi_initialized()
        error("MPI has not been initialized. Run `mpi_init()` to initialize MPI first.")
    end 
    if mpi_finalized()
        error("MPI has been finalized.")
    end
end

function _mpi_sum(a,root::Int64=0)
    mpi_check()
    mpisum_ = load_system_op("mpisum")
    a,root = convert_to_tensor(Any[a,root], [Float64,Int64])
    out = mpisum_(a,root)
    if !isnothing(length(a))
        return set_shape(out, (length(a),))
    else
        return out 
    end
end

"""
    mpi_sum(a::Union{Array{Float64}, Float64, PyObject}, root::Int64 = 0)

Sum `a` on the MPI processor `root`.
"""
function mpi_sum(a::Union{Array{Float64}, Float64, PyObject}, root::Int64 = 0)
    a = convert_to_tensor(a, dtype = Float64)
    if length(size(a))==0
        a = reshape(a, (1,))
        out = _mpi_sum(a, root)
        return squeeze(out)
    elseif length(size(a))==1
        return _mpi_sum(a, root)
    elseif nothing in size(a)
        error("The shape of 1st input $(size(a)) contains nothing. mpi_sum is not able to determine
        the output shape.")
    else 
        s = size(a)
        a = reshape(a, (-1,))
        out = _mpi_sum(a, root)
        return reshape(out, s)
    end
end

function _mpi_bcast(a,root::Int64=0)
    mpi_check()
    mpibcast_ = load_system_op("mpibcast")
    a,root = convert_to_tensor(Any[a,root], [Float64,Int64])
    out = mpibcast_(a,root)
    if !isnothing(length(a))
        return set_shape(out, (length(a),))
    else
        return out 
    end
end

"""
    mpi_bcast(a::Union{Array{Float64}, Float64, PyObject}, root::Int64 = 0)

Broadcast `a` from processor `root` to all other processors.
"""
function mpi_bcast(a::Union{Array{Float64}, Float64, PyObject}, root::Int64 = 0)
    a = convert_to_tensor(a, dtype = Float64)
    if length(size(a))==0
        a = reshape(a, (1,))
        out = _mpi_bcast(a, root)
        return squeeze(out)
    elseif length(size(a))==1
        return _mpi_bcast(a, root)
    elseif nothing in size(a)
        error("The shape of 1st input $(size(a)) contains nothing. mpi_bcast is not able to determine
        the output shape.")
    else 
        s = size(a)
        a = reshape(a, (-1,))
        out = _mpi_bcast(a, root)
        return reshape(out, s)
    end
end

function _mpi_send(a,dest::Int64,tag::Int64=0)
    mpisend_ = load_system_op("mpisend")
    a,dest,tag = convert_to_tensor(Any[a,dest,tag], [Float64,Int64,Int64])
    out = mpisend_(a,dest,tag)
    if !isnothing(length(a))
        return set_shape(out, (length(a),))
    else
        return out 
    end
end

"""
    mpi_send(a::Union{Array{Float64}, Float64, PyObject}, dest::Int64,root::Int64 = 0)

Sends `a` to processor `dest`. `a` itself is returned so that the send action can be added to the computational graph.
"""
function mpi_send(a::Union{Array{Float64}, Float64, PyObject}, dest::Int64,root::Int64 = 0)
    a = convert_to_tensor(a, dtype = Float64)
    if length(size(a))==0
        a = reshape(a, (1,))
        out = _mpi_send(a,dest, root)
        return squeeze(out)
    elseif length(size(a))==1
        return _mpi_send(a, dest, root)
    elseif nothing in size(a)
        error("The shape of 1st input $(size(a)) contains nothing. mpi_send is not able to determine
        the output shape.")
    else 
        s = size(a)
        a = reshape(a, (-1,))
        out = _mpi_send(a, dest, root)
        return reshape(out, s)
    end
end


function _mpi_recv(a,src::Int64,tag::Int64=0)
    mpirecv_ = load_system_op("mpirecv")
    a,src,tag = convert_to_tensor(Any[a,src,tag], [Float64,Int64,Int64])
    mpirecv_(a,src,tag)
end

"""
    mpi_recv(a::Union{Array{Float64}, Float64, PyObject}, src::Int64, tag::Int64 = 0)

Receives an array from processor `src`. `mpi_recv` requires an input for gradient backpropagation. 
Typically we can write

```julia
r = mpi_rank()
a = constant(Float64(r))
if r==1
    a = mpi_send(a, 0)
end
if r==0
    a = mpi_recv(a, 1)
end
```

Then `a=1` on both processor 0 and processor 1.
"""
function mpi_recv(a::Union{Array{Float64}, Float64, PyObject}, src::Int64, tag::Int64 = 0)
    a = convert_to_tensor(a, dtype = Float64)
    if length(size(a))==0
        a = reshape(a, (1,))
        out = _mpi_recv(a, src, tag)
        return squeeze(out)
    elseif length(size(a))==1
        return _mpi_recv(a, src, tag)
    elseif nothing in size(a)
        error("The shape of 1st input $(size(a)) contains nothing. mpi_recv is not able to determine
        the output shape.")
    else 
        s = size(a)
        a = reshape(a, (-1,))
        out = _mpi_recv(a, src, tag)
        return reshape(out, s)
    end
end

"""
    mpi_sendrecv(a::Union{Array{Float64}, Float64, PyObject}, dest::Int64, src::Int64, tag::Int64=0)    

A convenient wrapper for `mpi_send` followed by `mpi_recv`.
"""
function mpi_sendrecv(a::Union{Array{Float64}, Float64, PyObject}, dest::Int64, src::Int64, tag::Int64=0)
    r = mpi_rank()
    @assert src != dest
    if r==src 
        a = mpi_send(a, dest, tag)
    elseif r==dest 
        a = mpi_recv(a, src, tag)
    end
    a 
end

"""
    mpi_gather(u::Union{Array{Float64, 1}, PyObject})

Gathers all the vectors from different processes to the root process. The function returns 
a long vector which concatenates of local vectors in the order of process IDs. 
"""
function mpi_gather(u::Union{Array{Float64, 1}, PyObject})
    mpigather_ = load_system_op("mpigather")
    u = convert_to_tensor(Any[u], [Float64]); u = u[1]
    out = mpigather_(u)
    set_shape(out, (mpi_size()*length(u)))
end

function load_plugin_MPITensor()
    if !isfile("$(ADCME.LIBDIR)/libHYPRE.so")
        scriptpath = joinpath(@__DIR__, "..", "deps", "install_hypre.jl")
        include(scriptpath)
    end
    if Sys.isapple()
        oplibpath = joinpath(@__DIR__, "..", "deps", "Plugin", "MPITensor", "build", "libMPITensor.dylib")
    elseif Sys.iswindows()
        oplibpath = joinpath(@__DIR__, "..", "deps", "Plugin", "MPITensor", "build", "MPITensor.dll")
    else 
        oplibpath = joinpath(@__DIR__, "..", "deps", "Plugin", "MPITensor", "build", "libMPITensor.so")
    end
    if !isfile(oplibpath)
        PWD = pwd()
        cd(joinpath(@__DIR__, "..", "deps", "Plugin", "MPITensor"))
        if !isdir("build")
        mkdir("build")
        end
        cd("build")
        cmake()
        make()
        cd(PWD)
    end
    oplibpath
end

function load_plugin_MPIHaloExchange()
    if Sys.isapple()
        oplibpath = joinpath(@__DIR__, "..", "deps", "Plugin", "MPIHaloExchange", "build", "libHaloExchangeTwoD.dylib")
    elseif Sys.iswindows()
        oplibpath = joinpath(@__DIR__, "..", "deps", "Plugin", "MPIHaloExchange", "build", "HaloExchangeTwoD.dll")
    else 
        oplibpath = joinpath(@__DIR__, "..", "deps", "Plugin", "MPIHaloExchange", "build", "libHaloExchangeTwoD.so")
    end
    if !isfile(oplibpath)
        PWD = pwd()
        cd(joinpath(@__DIR__, "..", "deps", "Plugin", "MPIHaloExchange"))
        if !isdir("build")
        mkdir("build")
        end
        cd("build")
        cmake()
        make()
        cd(PWD)
    end
    oplibpath
end

function mpi_halo_exchange_(oplibpath, u,fill_value,m,n)
    halo_exchange_two_d_ = load_op_and_grad(oplibpath,"halo_exchange_two_d")
    u,fill_value,m,n = convert_to_tensor(Any[u,fill_value,m,n], [Float64,Float64,Int64,Int64])
    out = halo_exchange_two_d_(u,fill_value,m,n)
    set_shape(out, (size(u,1)+2, size(u,2)+2))
end

function mpi_create_matrix(oplibpath, indices,values,ilower,iupper)
    mpi_create_matrix_ = load_op_and_grad(oplibpath,"mpi_create_matrix", multiple=true)
    indices,values,ilower,iupper = convert_to_tensor(Any[indices,values,ilower,iupper], [Int64,Float64,Int64,Int64])
    mpi_create_matrix_(indices,values,ilower,iupper)
end

function mpi_get_matrix(oplibpath, rows,ncols,cols,ilower,iupper,values, N)
    mpi_get_matrix_ = load_op_and_grad(oplibpath,"mpi_get_matrix", multiple=true)
    rows,ncols,cols,ilower_,iupper_,values = convert_to_tensor(Any[rows,ncols,cols,ilower,iupper,values], [Int32,Int32,Int32,Int64,Int64,Float64])
    indices, vals = mpi_get_matrix_(rows,ncols,cols,ilower_,iupper_,values)
    SparseTensor(tf.SparseTensor(indices, vals, (iupper-ilower+1, N)), false)
end


function mpi_tensor_solve(oplibpath, rows,ncols,cols,values,rhs,ilower,iupper,solver,printlevel)
    mpi_tensor_solve_ = load_op_and_grad(oplibpath,"mpi_tensor_solve")
    rows,ncols,cols,values,rhs,ilower,iupper,printlevel = convert_to_tensor(Any[rows,ncols,cols,values,rhs,ilower,iupper,printlevel], [Int32,Int32,Int32,Float64,Float64,Int64,Int64,Int64])
    mpi_tensor_solve_(rows,ncols,cols,values,rhs,ilower,iupper,solver,printlevel)
end

@doc raw"""
    mutable struct mpi_SparseTensor
        rows::PyObject 
        ncols::PyObject
        cols::PyObject 
        values::PyObject 
        ilower::Int64 
        iupper::Int64 
        N::Int64
        oplibpath::String
    end

A structure to hold local data of a sparse matrix. The global matrix is assumed to be a $M\times N$ square matrix. 
The current processor owns rows from `ilower` to `iupper` (inclusive). The data is specified by 

- `rows`: the row indices that the local matrix owns. Each index should appear only once. 
- `ncols`: for each row index, the number of nonzero entries in this row 
- `cols`: the column indices 
- `values`: the nonzero entries corresponding to `cols`
- `oplibpath`: the backend library (returned by `ADCME.load_plugin_MPITensor`)

All data structure are 0-based. Note if we work with a linear solver, $M=N$.

For example, consider the sparse matrix 

```
[  1 0 0 1  ]
[  0 1 2 1  ]
```

We have 

```julia
rows = Int32[0;1]
ncols = Int32[2;3]
cols = Int32[0;3;5;6;7]
values = [1.;1.;1.;2.;1.]
iupper = ilower + 2
```
"""
mutable struct mpi_SparseTensor
    rows::PyObject 
    ncols::PyObject
    cols::PyObject 
    values::PyObject 
    ilower::Int64 
    iupper::Int64 
    N::Int64
    oplibpath::String
end

function Base.:show(io::IO, sp::mpi_SparseTensor)
    if isnothing(length(sp.values))
        len = "?"
    else 
        len = length(sp.values)
    end
    print("mpi_SparseTensor($(sp.iupper - sp.ilower + 1), $(sp.N)), range = [$(sp.ilower), $(sp.iupper)], nnz = $(len)")
end

function mpi_SparseTensor(indices::PyObject, values::PyObject, ilower::Int64, iupper::Int64, N::Int64)
    @assert ilower >=0
    @assert ilower <= iupper 
    @assert iupper <= N
    oplibpath = load_plugin_MPITensor()
    rows, ncols, cols, out = mpi_create_matrix(oplibpath, indices,values,ilower,iupper)
    mpi_SparseTensor(rows, ncols, cols, out, ilower, iupper, N, oplibpath)
end 

function mpi_SparseTensor(rows::Union{Array{Int32,1}, PyObject}, ncols::Union{Array{Int32,1}, PyObject}, cols::Union{Array{Int32,1}, PyObject},
     vals::Union{Array{Float64,1}, PyObject}, ilower::Int64, iupper::Int64, N::Int64)
    @assert ilower >=0
    @assert ilower <= iupper 
    @assert iupper <= N
    oplibpath = load_plugin_MPITensor()
    mpi_SparseTensor(rows, ncols, cols, vals, ilower, iupper, N, oplibpath)
end

"""
    mpi_SparseTensor(sp::Union{SparseTensor, SparseMatrixCSC{Float64,Int64}}, 
        ilower::Union{Int64, Missing} = missing,
        iupper::Union{Int64, Missing} = missing)

Constructing `mpi_SparseTensor` from a `SparseTensor` or a sparse Array.
"""
function mpi_SparseTensor(sp::Union{SparseTensor, SparseMatrixCSC{Float64,Int64}}, 
    ilower::Union{Int64, Missing} = missing,
    iupper::Union{Int64, Missing} = missing)
    sp = constant(sp)
    ilower = coalesce(ilower, 0)
    iupper = coalesce(iupper, size(sp, 1)-1)
    N = size(sp, 2)
    mpi_SparseTensor(sp.o.indices, sp.o.values, ilower, iupper, N)
end

function LinearAlgebra.:\(sp::mpi_SparseTensor, b::Union{Array{Float64, 1}, PyObject})
    b = constant(b)
    out = mpi_tensor_solve(sp.oplibpath, sp.rows,sp.ncols,
                sp.cols,sp.values,b,
                sp.ilower,sp.iupper,options.mpi.solver, options.mpi.printlevel)
    set_shape(out, (sp.iupper-sp.ilower+1,))
end

function SparseTensor(sp::mpi_SparseTensor)
    mpi_get_matrix(sp.oplibpath, sp.rows,sp.ncols,sp.cols,sp.ilower,sp.iupper,sp.values, sp.N)
end

function Base.:run(sess::PyObject, sp::mpi_SparseTensor)
    run(sess, SparseTensor(sp))
end


@doc raw"""
    mpi_halo_exchange(u::Union{Array{Float64, 2}, PyObject},m::Int64,n::Int64; fill_value::Float64 = 0.0)

Perform Halo exchnage on `u` (a $k \times k$ matrix). The output has a shape $(k+2)\times (k+2)$
"""
function mpi_halo_exchange(u::Union{Array{Float64, 2}, PyObject},m::Int64,n::Int64; fill_value::Float64 = 0.0)
    @assert size(u,1)==size(u,2)
    oplibpath = load_plugin_MPIHaloExchange()
    mpi_halo_exchange_(oplibpath, u, fill_value, m, n)
end