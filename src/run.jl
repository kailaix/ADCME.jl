import Base:run

export
run,
Session,
init,
run_profile,
save_profile

"""
    Session(args...; kwargs...)

Create an ADCME session. By default, ADCME will take up all the GPU resources at the start. If you want the GPU usage to grow on a need basis,
before starting ADCME, you need to set the environment variable via

```julia
ENV["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
```

# Configuration

Session accepts some runtime optimization configurations 

- `intra`: Number of threads used within an individual op for parallelism
- `inter`: Number of threads used for parallelism between independent operations.
- `CPU`: Maximum number of CPUs to use. 
- `GPU`: Maximum number of GPU devices to use
- `soft`: Set to True/enabled to facilitate operations to be placed on CPU instead of GPU

!!! note 
    `CPU` limits the number of CPUs being used, not the number of cores or threads.
"""
function Session(args...; kwargs...)
    

    kwargs_ = Dict{Symbol, Any}()
    if haskey(kwargs, :intra)
        kwargs_[:intra_op_parallelism_threads] = kwargs[:intra]
    end
    if haskey(kwargs, :inter)
        kwargs_[:inter_op_parallelism_threads] = kwargs[:intra]
    end
    if haskey(kwargs, :soft)
        kwargs_[:allow_soft_placement] = kwargs[:soft]
    end
    if haskey(kwargs, :CPU) || haskey(kwargs, :GPU)
        cnt = Dict{String, Int64}()
        if haskey(kwargs, :CPU)
            cnt["CPU"] = kwargs[:CPU]
        end
        if haskey(kwargs, :GPU)
            cnt["GPU"] = kwargs[:GPU]
        end
        kwargs_[:device_count] = cnt
    end

    if haskey(kwargs, :config)
        sess = tf.compat.v1.Session(args...;kwargs...)
    else
        config = tf.ConfigProto(;kwargs_...)
        sess = tf.compat.v1.Session(config = config)
    end
    
    STORAGE["session"] = sess 
    return sess 
end

function Base.:run(sess::PyObject, fetches::Union{PyObject, Array{PyObject}, Array{Any}, Tuple}, args::Pair{PyObject, <:Any}...; kwargs...)
    local ret 
    if length(args)>0
        ret = sess.run(fetches, feed_dict = Dict(args))
    else
        ret = sess.run(fetches; kwargs...)
    end 
    if isnothing(ret)
        return nothing
    elseif isa(ret, Array) && size(ret)==()
        return ret[1]
    else
        return ret
    end
end

function global_variables_initializer()
    tf.compat.v1.global_variables_initializer()
end

function init(o::PyObject)
    run(o, global_variables_initializer())
end

"""
    run_profile(args...;kwargs...)

Runs the session with tracing information.
"""
function run_profile(args...;kwargs...)
    global run_metadata
    options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()
    run(args...;options=options, run_metadata=run_metadata, kwargs...)
end

"""
    save_profile(filename::String="default_timeline.json")

Save the timeline information to file `filename`. 
- Open Chrome and navigate to chrome://tracing
- Load the timeline file
"""
function save_profile(filename::String="default_timeline.json"; kwargs...)
    timeline = pyimport("tensorflow.python.client.timeline")
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format(;kwargs...)
    open(filename,"w") do io
        write(io, chrome_trace)
    end
    @info "Timeline information saved in $filename
- Open Chrome and navigate to chrome://tracing
- Load the timeline file"
end