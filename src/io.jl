using MAT
export 
save,
load,
diary,
scalar,
writepb


pybytes(b) = PyObject(
    ccall(PyCall.@pysym(PyCall.PyString_FromStringAndSize),
                                             PyPtr, (Ptr{UInt8}, Int),
                                             b, sizeof(b)))

"""
`save` saves the values of tensors in the session `sess`. The result is written into `file` as a dictionary. 
see also `load`
"""
function save(sess::PyObject, file::String, vars::Union{PyObject, Nothing, Array{PyObject}}=nothing, args...; kwargs...)
    if vars==nothing
        vars = get_collection(TRAINABLE_VARIABLES)
    elseif isa(vars, PyObject)
        vars = [vars]
    end
    d = Dict{String, Any}()
    vals = run(sess, vars, args...;kwargs...)
    for i = 1:length(vars)
        name = replace(vars[i].name, ":"=>"colon")
        name = replace(name, "/"=>"backslash")
        d[name] = vals[i]
    end
    matwrite(file, d)
end

function save(sess::PyObject, vars::Union{PyObject, Nothing, Array{PyObject}}=nothing, args...; kwargs...)
    if vars==nothing
        vars = get_collection(TRAINABLE_VARIABLES)
    elseif isa(vars, PyObject)
        vars = [vars]
    end
    d = Dict{String, Any}()
    vals = run(sess, vars, args...;kwargs...)
    for i = 1:length(vars)
        name = replace(vars[i].name, ":"=>"colon")
        name = replace(name, "/"=>"backslash")
        d[name] = vals[i]
    end
    d
end

"""
`load` loads the values of tensors to the session `sess`. 
see also `save`
"""
function load(sess::PyObject, file::String, vars::Union{PyObject, Nothing, Array{PyObject}}=nothing, args...; kwargs...)
    if vars==nothing
        vars = get_collection(TRAINABLE_VARIABLES)
    elseif isa(vars, PyObject)
        vars = [vars]
    end
    d = matread(file)
    ops = PyObject[]
    for i = 1:length(vars)
        name = replace(vars[i].name, ":"=>"colon")
        name = replace(name,  "/"=>"backslash")
        if !(name in keys(d))
            @warn "$(vars[i].name) not found in the file, skipped"
        else
            if occursin("bias", name) && isa(d[name], Number)
                d[name] = [d[name]]
            end
            push!(ops, assign(vars[i], d[name]))
        end
    end
    run(sess, ops, args...; kwargs...)
end

function load(sess::PyObject, d::Dict, vars::Union{PyObject, Nothing, Array{PyObject}}=nothing, args...; kwargs...)
    if vars==nothing
        vars = get_collection(TRAINABLE_VARIABLES)
    elseif isa(vars, PyObject)
        vars = [vars]
    end
    ops = PyObject[]
    for i = 1:length(vars)
        name = replace(vars[i].name, ":"=>"colon")
        name = replace(name,  "/"=>"backslash")
        if !(name in keys(d))
            @warn "$(vars[i].name) not found in the file, skipped"
        else
            push!(ops, assign(vars[i], d[name]))
        end
    end
    run(sess, ops, args...; kwargs...)
end

function diary(logdir::String)
    if logdir[1]!='/' && logdir[1]!="."
        logdir = "./"*logdir
    end
    try
        run(`rm -rf $logdir`)
    catch
        run(`rm -rf $logdir/*`)
    end
    println("""** Run the following command in terminal **""")
    printstyled("""tensorboard --logdir="$(abspath(logdir))" --port 6006\n""", color=:cyan, bold = true)
    tf.summary.FileWriter(logdir, tf.get_default_graph())
end

function scalar(o::PyObject, name::String)
    tf.summary.scalar(name, o)
end

function Base.:write(writer::PyObject, cnt::Union{String, Array{String}}, step::Int64)
    if isa(cnt, String)
        writer.add_summary(pybytes(cnt), step)
    else
        for c in cnt 
            writer.add_summary(pybytes(c), step)
        end
    end
end

function writepb(writer::PyObject, sess::PyObject)
    output_path = writer.get_logdir()
    tf.io.write_graph(sess.graph_def, output_path, "model.pb")
    return
end

