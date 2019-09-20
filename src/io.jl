using MAT
export 
save,
load,
Diary,
scalar,
writepb,
psave,
pload,
activate


pybytes(b) = PyObject(
    ccall(PyCall.@pysym(PyCall.PyString_FromStringAndSize),
                                             PyPtr, (Ptr{UInt8}, Int),
                                             b, sizeof(b)))

"""
    psave(o::PyObject, file::String)

Saves a Python objection `o` to `file`.
See also [`pload`](@ref)
"""                                             
function psave(o::PyObject, file::String)
    f = open(file, "w")
    pickle.dump(o, f)
    close(f)
end

"""
    pload(file::String)

Loads a Python objection from `file`.
See also [`psave`](@ref)
"""                
function pload(file::String)
    r = nothing
    @pywith pybuiltin("open")(file,"rb") as f begin
        r = pickle.load(f)
    end
    return r
end

"""
    save(sess::PyObject, file::String, vars::Union{PyObject, Nothing, Array{PyObject}}=nothing, args...; kwargs...)

Saves the values of `vars` in the session `sess`. The result is written into `file` as a dictionary. If `vars` is nothing, it saves all the trainable variables.
See also [`save`](@ref), [`load`](@ref)
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
    load(sess::PyObject, file::String, vars::Union{PyObject, Nothing, Array{PyObject}}=nothing, args...; kwargs...)

Loads the values of variables to the session `sess` from the file `file`. If `vars` is nothing, it loads values to all the trainable variables.
See also [`save`](@ref), [`load`](@ref)
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


mutable struct Diary
    writer::PyObject
    tdir::String
end

"""
    Diary(suffix::Union{String, Nothing}=nothing)

Creates a diary at a temporary directory path. It returns a writer and the corresponding directory path
"""
function Diary(suffix::Union{String, Nothing}=nothing)
    tdir = mktempdir()
    printstyled("tensorboard --logdir=\"$(tdir)\" --port 0\n", color=:blue)
    Diary(tf.summary.FileWriter(tdir, tf.get_default_graph(),filename_suffix=suffix), tdir)
end

"""
    save(sw::Diary, dirp::String)

Saves [`Diary`](@ref) to `dirp`.
"""
function save(sw::Diary, dirp::String)
    cp(sw.tdir, dirp, force=true)
end

"""
    load(sw::Diary, dirp::String)

Loads [`Diary`](@ref) from `dirp`.
"""
function load(sw::Diary,dirp::String)
    sw.writer = tf.summary.FileWriter(dirp, tf.get_default_graph())
    sw.tdir = dirp
end

"""
    activate(sw::Diary, port::Int64=6006)

Running [`Diary`](@ref) at http://localhost:port.
"""
function activate(sw::Diary, port::Int64=0)
    printstyled("tensorboard --logdir=\"$(sw.tdir)\" --port $port\n", color=:blue)
    run(`tensorboard --logdir="$(sw.tdir)" --port $port &`)
end

"""
    scalar(o::PyObject, name::String)

Returns a scalar summary object.
"""
function scalar(o::PyObject, name::Union{String,Missing}=missing)
    if ismissing(name)
        name = tensorname(o)
    end
    tf.summary.scalar(name, o)
end

"""
    write(sw::Diary, step::Int64, cnt::Union{String, Array{String}})

Writes to [`Diary`](@ref).
"""
function Base.:write(sw::Diary, step::Int64, cnt::Union{String, Array{String}})
    if isa(cnt, String)
        sw.writer.add_summary(pybytes(cnt), step)
    else
        for c in cnt 
            sw.writer.add_summary(pybytes(c), step)
        end
    end
end

function writepb(writer::PyObject, sess::PyObject)
    output_path = writer.get_logdir()
    tf.io.write_graph(sess.graph_def, output_path, "model.pb")
    return
end

