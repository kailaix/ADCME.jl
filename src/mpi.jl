export mpi_bcast, mpi_init, mpi_recv, mpi_send, 
    mpi_sendrecv, mpi_sum,  mpi_finalize, mpi_initialized,
    mpi_finalized, mpi_rank, mpi_size

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

function mpi_check()
    if !mpi_initialized()
        error("MPI has not been initialized.")
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
    elseif length(size(a))==0
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
    elseif length(size(a))==0
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
    elseif length(size(a))==0
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
    elseif length(size(a))==0
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