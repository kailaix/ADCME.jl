export mpi_bcast, mpi_init, mpi_recv, mpi_send, 
    mpi_sendrecv, mpi_sum,  mpi_finalize, mpi_initialized,
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

Sync `message` across all MPI processors.
"""
function mpi_sync!(message::Array{Int64,1}, root::Int64 = 0)
    mpi_check()
    @eval ccall((:mpi_sync, $LIBADCME), Cvoid, (Ptr{Clonglong}, Cint, Cint), $message, Int32(length($message)), Int32($root))
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
    scriptpath = joinpath(@__DIR__, "..", "deps", "install_hypre.jl")
    include(scriptpath)
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


function mpi_tensor_solve(oplibpath, rows,ncols,cols,values,rhs,ilower,iupper,solver = "BoomerAMG",printlevel = 2)
    mpi_tensor_solve_ = load_op_and_grad(oplibpath,"mpi_tensor_solve")
    rows,ncols,cols,values,rhs,ilower,iupper,printlevel = convert_to_tensor(Any[rows,ncols,cols,values,rhs,ilower,iupper,printlevel], [Int32,Int32,Int32,Float64,Float64,Int64,Int64,Int64])
    mpi_tensor_solve_(rows,ncols,cols,values,rhs,ilower,iupper,solver,printlevel)
end


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
    set_shape(out, (length(b),))
end

function SparseTensor(sp::mpi_SparseTensor)
    mpi_get_matrix(sp.oplibpath, sp.rows,sp.ncols,sp.cols,sp.ilower,sp.iupper,sp.values, sp.N)
end

function Base.:run(sess::PyObject, sp::mpi_SparseTensor)
    run(sess, SparseTensor(sp))
end