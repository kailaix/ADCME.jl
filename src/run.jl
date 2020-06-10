import Base:run

export
run,
Session,
init,
run_profile,
save_profile

"""
    Session(args...;allow_growth::Bool = false, kwargs...)

Create an ADCME session. If `allow_growth` is true, when using GPU, ADCME will not take up all the GPU resources at the start, 
but instead request memory on a need basis. 
"""
function Session(args...;allow_growth::Bool = false, kwargs...)
    if allow_growth
        ENV["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    else
        ENV["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
    end
    sess = tf.compat.v1.Session(args...;kwargs...)
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