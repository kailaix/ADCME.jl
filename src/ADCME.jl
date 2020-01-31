
__precompile__(true)
module ADCME

    export tf,
    AUTO_REUSE,
    GLOBAL_VARIABLES,
    TRAINABLE_VARIABLES,
    UPDATE_OPS
    
    using PyCall
    using Random
    using LinearAlgebra
    using Conda
    import Optim
    using SparseArrays

    tf = PyNULL()
    DTYPE = Dict{Type, PyObject}()
    # a list of custom operators 
    COLIB = Dict{String, Tuple{String, String, String, Bool}}()

    libSuffix = Sys.isapple() ? "dylib" : (Sys.islinux() ? "so" : "dll")
    
    if isfile("$(@__DIR__)/../deps/deps.jl"); include("$(@__DIR__)/../deps/deps.jl"); end
    run_metadata = nothing
    
    
    function __init__()
        # install_custom_op_dependency() # always install dependencies
        global AUTO_REUSE, GLOBAL_VARIABLES, TRAINABLE_VARIABLES, UPDATE_OPS, DTYPE, COLIB
py"""
import warnings;warnings.filterwarnings('ignore')
"""
        copy!(tf, pyimport("tensorflow"))
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

        colibs = readlines("$(@__DIR__)/../deps/CustomOps/default_formulas.txt")
        for c in colibs
            push!(COLIB, eval(Meta.parse(c)))
        end
        if isfile("$(@__DIR__)/../deps/CustomOps/formulas.txt")
            colibs = readlines("$(@__DIR__)/../deps/CustomOps/formulas.txt")
            for c in colibs
                push!(COLIB, eval(Meta.parse(c)))
            end
        end
    end

    include("core.jl")
    include("io.jl")
    include("optim.jl")
    include("run.jl")
    include("variable.jl")
    include("ops.jl")
    include("layers.jl")
    include("extra.jl")
    include("sparse.jl")
    include("random.jl")
    include("gan.jl")
    include("ot.jl")
    include("ode.jl")
end

