export
reset_default_graph,
get_collection,
enable_eager_execution,
control_dependencies,
has_gpu,
while_loop,
if_else,
stop_gradient,
independent,
tensor,
tensorname,
has_mpi,
get_mpi,
get_mpirun,
@cpu, 
@gpu 

# only for eager eager execution
enable_eager_execution() = tf.enable_eager_execution()
Base.:values(o::PyObject) = o.numpy()

"""
    reset_default_graph()

Resets the graph by removing all the operators. 
"""
function reset_default_graph()
    global STORAGE
    tf.compat.v1.reset_default_graph()
    STORAGE = Dict{String, Any}()
    nothing
end
"""
    get_collection(name::Union{String, Missing})

Returns the collection with name `name`. If `name` is `missing`, returns all the trainable variables.
"""
function get_collection(name::Union{String, Missing}=missing)
    if !ismissing(name) && (name in [TRAINABLE_VARIABLES, UPDATE_OPS])
        return tf.compat.v1.get_collection(name)
    end
    if ismissing(name)
        res = tf.compat.v1.get_collection(TRAINABLE_VARIABLES)
    else
        res = []
        vs = tf.compat.v1.get_collection(TRAINABLE_VARIABLES)
        rname = @eval @r_str $name
        for v in vs 
            if occursin(rname, v.name)
                push!(res, v)
            end
        end
    end
    return unique(res)
end


"""
    tensor(s::String)

Returns the tensor with name `s`. See [`tensorname`](@ref).
"""
function tensor(s::String)
    tf.get_default_graph().get_tensor_by_name(s)
end

"""
    tensorname(o::PyObject)

Returns the name of the tensor. See [`tensor`](@ref).
"""
function tensorname(o::PyObject)
    o.name
end

function jlargs(kwargs)
    kwargs = Dict{Any, Any}(kwargs)
    if :axis in keys(kwargs)
        @error("axis is not a valid keyword, using dims instead (base = 1)")
    end
    if :dtype in keys(kwargs)
        kwargs[:dtype] = DTYPE[kwargs[:dtype]]
    end
    if :dims in keys(kwargs)
        kwargs[:axis] = kwargs[:dims] .- 1
        if isa(kwargs[:axis], Array)
            kwargs[:axis] = Tuple(kwargs[:axis])
        end
        delete!(kwargs, :dims)
    end
    if :colocate in keys(kwargs)
        kwargs[:colocate_gradients_with_ops] = kwargs[:colocate]
        delete!(kwargs, :colocate)
    end
    kwargs
end

# control_dependencies can be used to fix the memory problem
# https://stackoverflow.com/questions/39350164/tensorflow-parallel-for-loop-results-in-out-of-memory

"""
    control_dependencies(f, ops::Union{Array{PyObject}, PyObject})

Executes all operations in `ops` before any operations _created_ inside the block. 
```julia
op1 = tf.print("print op1")
op3 = tf.print("print op3")
control_dependencies(op1) do
    global op2 = tf.print("print op2")
end
run(sess, [op2,op3])
```
In this example, `op1` must be executed before `op2`. But there is no guarantee when `op3` will be executed. 
There are several possible outputs of the program such as

```julia-repl
print op3
print op1
print op2
```
or 
```
print op1
print op3
print op2
```
"""
function control_dependencies(f, ops::Union{Tuple, Array{PyObject}, PyObject})
    if isa(ops, PyObject)
        ops = [ops]
    end
    @pywith tf.control_dependencies(ops) begin
        f()
    end
end

"""
    bind(op::PyObject, ops...)

Adding operations `ops` to the dependencies of `op`. `ops` are guaranteed to be executed **before** `op`.
The function is useful when we want to execute `ops` but `ops` is not 
in the dependency of the final output. For example, if we want to print `i` each time `i` is evaluated
```julia
i = constant(1.0)
op = tf.print(i)
i = bind(i, op)
```
"""
function Base.:bind(op::PyObject, ops...)
    local op1
    control_dependencies(ops) do 
        op1 = tf.identity(op)
    end
    return op1
end

@doc raw"""
    while_loop(condition::Union{PyObject,Function}, body::Function, loop_vars::Union{PyObject, Array{Any}, Array{PyObject}};
        parallel_iterations::Int64=10, kwargs...)

Loops over `loop_vars` while `condition` is true. This operator only creates one extra node to mark the loops in the computational graph.

# Example

The following script computes 
```math
\sum_{i=1}^{10} i
```

```julia
function condition(i, ta)
    i <= 10
end
function body(i, ta)
    u = read(ta, i-1)
    ta = write(ta, i, u+1)
    i+1, ta
end
ta = TensorArray(10)
ta = write(ta, 1, constant(1.0))
i = constant(2, dtype=Int32)
_, out = while_loop(condition, body, [i, ta])
summation = stack(out)[10]
```
"""
function while_loop(condition::Union{PyObject,Function}, body::Function, loop_vars::Union{PyObject, Array{Any}, Array{PyObject}};
        parallel_iterations::Int64=10, kwargs...)
    # @warn "TensorArray must be initialized (writedown at index 1) outside" maxlog=1
    if isa(loop_vars, PyObject)
        lv = [loop_vars]
    else
        lv = loop_vars
    end
    if get_dtype(loop_vars[1])!=Int32
        error("Loop index must be Int32, got $(get_dtype(loop_vars[1]))")
    end

    res = tf.while_loop(condition, body, loop_vars=lv; parallel_iterations=parallel_iterations, kwargs...)

    if isa(loop_vars, PyObject)
        return res[1]
    else
        return res
    end
end

function if_else_v1(condition::Union{PyObject}, fn1, fn2, args...;kwargs...)
    fn1_ = ifelse(isa(fn1, Function), fn1, ()->fn1)
    fn2_ = ifelse(isa(fn2, Function), fn2, ()->fn2)
    tf.cond(condition, fn1_, fn2_, args...;kwargs...)
end 

function if_else_v2(condition::PyObject, fn1::Union{Nothing, PyObject, Array}, 
        fn2::Union{Nothing, PyObject, Array})
    fn1 = convert_to_tensor(fn1)
    fn2 = convert_to_tensor(fn2)
    tf.compat.v2.where(condition, fn1, fn2) 
end 


"""
    if_else(condition::Union{PyObject,Array,Bool}, fn1, fn2, args...;kwargs...)

- If `condition` is a scalar boolean, it outputs `fn1` or `fn2` (a function with no input argument or a tensor) based on whether `condition` is true or false.
- If `condition` is a boolean array, if returns `condition .* fn1 + (1 - condition) .* fn2`

!!! info 
    If you encounter an error like this:
    ```
    tensorflow.python.framework.errors_impl.InvalidArgumentError: Retval[0] does not have value
    ```
    It's probably that your code within `if_else` is not valid. 
"""
function if_else(condition::Union{PyObject,Array,Bool}, fn1, fn2, args...;kwargs...)
    if isa(condition, Array) || isa(condition, Bool)
        condition = convert_to_tensor(condition)
    end
    if isa(condition, Function) || (eltype(condition)<:Bool && length(size(condition))==0)
        if_else_v1(condition, fn1, fn2, args...;kwargs...)
    else
        if_else_v2(condition, fn1, fn2)
    end
end

"""
    has_gpu()

Checks if GPU is available.

!!! note
    ADCME will use GPU automatically if GPU is available. To disable GPU, set the environment variable `ENV["CUDA_VISIBLE_DEVICES"]=""` before importing ADCME 
"""
function has_gpu()
    s = tf.test.gpu_device_name()
    if length(s)==0
        return false
    else
        return true
    end
end

"""
    has_mpi(verbose::Bool = true)

Determines whether MPI is installed. 
"""
function has_mpi(verbose::Bool = true)
    if Sys.iswindows()
        if haskey(ENV, "MSMPI_INC") && haskey(ENV, "MSMPI_LIB64")
            return true 
        else 
            return false 
        end
    end
    if haskey(ENV, "MPI_INCLUDE_PATH") && haskey(ENV, "MPI_C_LIBRARIES")
        if !(isdir(ENV["MPI_INCLUDE_PATH"]) && "mpi.h" in readdir(ENV["MPI_INCLUDE_PATH"]))
            error("mpi.h is not found in ENV[\"MPI_INCLUDE_PATH\"] = $(ENV["MPI_INCLUDE_PATH"])")
        end
        if !isfile(ENV["MPI_C_LIBRARIES"])
            error("ENV[\"MPI_C_LIBRARIES\"]=$(ENV["MPI_C_LIBRARIES"]) does not exists.")
        end
        verbose && (@info "Use MPI libraries: $(ENV["MPI_C_LIBRARIES"])")
        return true 
    end 
    if isfile(get_library(joinpath(ADCME.LIBDIR, "mpi"))) && isfile(joinpath(ADCME.INCDIR, "mpi.h"))
        verbose && (@info "Use default MPI library (OpenMPI)")
        return true 
    end
    return false
end

""" 
    get_mpi()

Returns the MPI include directory and shared library.
"""
function get_mpi()
    if Sys.iswindows()
        if haskey(ENV, "MSMPI_INC") && haskey(ENV, "MSMPI_LIB64")
            return ENV["MSMPI_INC"], joinpath(ENV["MSMPI_LIB64"], "msmpi.lib")
        else 
            return false 
        end
    end
    if haskey(ENV, "MPI_INCLUDE_PATH") && haskey(ENV, "MPI_C_LIBRARIES")
        return ENV["MPI_INCLUDE_PATH"], ENV["MPI_C_LIBRARIES"]
    end 
    if isfile(get_library(joinpath(ADCME.LIBDIR, "mpi"))) && isfile(joinpath(ADCME.INCDIR, "mpi.h"))
        return ADCME.INCDIR, get_library(joinpath(ADCME.LIBDIR, "mpi"))
    end
    error("""MPI Library is not found. 
- On Windows, you can download Microsoft MPI (https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
- On Unix, you can install OpenMPI via `install_openmpi()`
""")
end

"""
    get_mpirun()

Returns the **default** mpirun executable. 
"""
function get_mpirun()
    if !has_mpi(false)
        error("""MPI Library is not found. 
- On Windows, you can download Microsoft MPI (https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
- On Unix, you can install OpenMPI via `install_openmpi()`
""")
    end
    if Sys.iswindows()
        return joinpath(ENV["MPI_BIN"], "mpiexec.exe")
    else
        if haskey(ENV, "MPI_INCLUDE_PATH") && haskey(ENV, "MPI_C_LIBRARIES")
            @warn("You are not using the default MPI. Trying to detect the executable...")
            mpirun = abspath(joinpath(ENV["MPI_INCLUDE_PATH"], "..", "bin", "mpirun"))
            if !isfile(mpirun)
                error("Failed.")
            else
                return mpirun 
            end
        end 

        mpirun =  joinpath(ADCME.BINDIR, "mpirun")
        if !isfile(mpirun)
            error("Failed.")
        end
        return mpirun 
    end
end

"""
    independent(o::PyObject, args...; kwargs...)

Returns `o` but when computing the gradients, the top gradients will not be back-propagated into dependent variables of `o`.
"""
independent(o::PyObject, args...; kwargs...) = stop_gradient(o, args...; kwargs...)

@deprecate stop_gradient independent
function stop_gradient(o::PyObject, args...;kwargs...)
    tf.stop_gradient(o, args...;kwargs...)
end

macro cpu(device_id, expr)
    device = "/cpu:"*string(device_id)
    quote  
        @pywith tf.device($device) begin 
            $(esc(expr))
        end
    end
end

macro cpu(expr)
    device = "/cpu:0"
    quote  
        @pywith tf.device($device) begin 
            $(esc(expr))
        end
    end
end

macro gpu(device_id, expr)
    device = "/gpu:"*string(device_id)
    quote  
        @pywith tf.device($device) begin 
            $(esc(expr))
        end
    end
end

macro gpu(expr)
    device = "/gpu:0"
    quote  
        @pywith tf.device($device) begin 
            $(esc(expr))
        end
    end
end