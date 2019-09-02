
__precompile__(true)
module ADCME

    export tf,
    AUTO_REUSE,
    GLOBAL_VARIABLES,
    TRAINABLE_VARIABLES,
    UPDATE_OPS
    
    using PyCall
    using Random

    tf = PyNULL()
    tfp = PyNULL()
    tfops = PyNULL()
    gradients_impl = PyNULL()
    DTYPE = Dict{Type, PyObject}()
    COFUNC = Dict{String, Union{Nothing,Function}}()
    COOK = false
    global AUTO_REUSE, GLOBAL_VARIABLES, TRAINABLE_VARIABLES, UPDATE_OPS
    function __init__()
        global AUTO_REUSE, GLOBAL_VARIABLES, TRAINABLE_VARIABLES, UPDATE_OPS, COFUNC, COOK, DTYPE
        copy!(tf, pyimport("tensorflow"))
        copy!(tfops, pyimport("tensorflow.python.framework.ops"))
        copy!(tfp, pyimport("tensorflow_probability"))
        copy!(gradients_impl, pyimport("tensorflow.python.ops.gradients_impl"))
        DTYPE = Dict(Float64=>tf.float64,
            Float32=>tf.float32,
            Int64=>tf.int64,
            Int32=>tf.int32,
            Bool=>tf.bool,
            ComplexF64=>tf.complex128,
            ComplexF32=>tf.complex64)
        AUTO_REUSE = tf.compat.v1.AUTO_REUSE
        GLOBAL_VARIABLES = tf.compat.v1.GraphKeys.GLOBAL_VARIABLES
        TRAINABLE_VARIABLES = tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES
        UPDATE_OPS = tf.compat.v1.GraphKeys.UPDATE_OPS
        CODIR = joinpath(@__DIR__, "../deps/CustomOps")
        COFUNC["sparse_solver"]=load_op_and_grad(joinpath(CODIR, "SparseSolver/build/libSparseSolver"), "sparse_solver")
        COOK = !any([v===nothing for (k,v) in COFUNC])
    end

    include("core.jl")
    include("io.jl")
    include("optim.jl")
    include("run.jl")
    include("variable.jl")
    include("ops.jl")
    include("layers.jl")
    include("datasets.jl")
    include("extra.jl")
    include("RBF.jl")
    include("sparse.jl")
    include("random.jl")
end
