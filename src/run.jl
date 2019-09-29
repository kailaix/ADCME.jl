import Base:run

export
run,
Session,
global_variables_initializer,
init

function Session(args...;kwargs...)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=true
    @suppress tf.compat.v1.Session(args...;config=config, kwargs...)
end

function Base.:run(o::PyObject, fetches::Union{Nothing, PyObject, Array{PyObject}, Array{Any}, Tuple}=nothing, args::Union{Nothing, PyObject, Array{PyObject}}...; kwargs...)
    o.run(fetches, args...; kwargs...)
end

function Base.:run(o::PyObject, fetches::Union{PyObject, Array{PyObject}, Array{Any}, Tuple}, args::Pair{PyObject, <:Any}...; kwargs...)
    o.run(fetches, feed_dict = Dict(args))
end

function Base.:run(o::PyObject, fetches::Union{PyObject, Array{PyObject,N} where N, Array{Any}, Tuple})
    o.run(fetches)
end

function global_variables_initializer()
    tf.compat.v1.global_variables_initializer()
end

function init(o::PyObject)
    run(o, global_variables_initializer())
end
