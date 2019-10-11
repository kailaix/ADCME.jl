import Base:run

export
run,
Session,
init,
run_profile,
save_profile

function Session(args...;kwargs...)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=true
    @suppress tf.compat.v1.Session(args...;config=config, kwargs...)
end

function Base.:run(o::PyObject, fetches::Union{PyObject, Array{PyObject}, Array{Any}, Tuple}, args::Pair{PyObject, <:Any}...; kwargs...)
    if length(args)>0
        o.run(fetches, feed_dict = Dict(args))
    else
        o.run(fetches; kwargs...)
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
function save_profile(filename::String="default_timeline.json")
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    open(filename,"w") do io
        write(io, chrome_trace)
    end
    @info "Timeline information saved in $filename
- Open Chrome and navigate to chrome://tracing
- Load the timeline file"
end