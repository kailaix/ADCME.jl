
__precompile__(true)
module ADCME

    export tf,
    tfp,
    AUTO_REUSE,
    GLOBAL_VARIABLES,
    TRAINABLE_VARIABLES,
    UPDATE_OPS
    
    using PyCall
    using Random
    using LinearAlgebra
    using SparseArrays
    using LibGit2
    using Libdl

    tf = PyNULL()
    tfp = PyNULL()

    #------------------------------------------------------------------------------------------
    # Global Storage 
    DTYPE = Dict{Type, PyObject}()
    COLIB = Dict{String, Tuple{String, String, String, Bool}}() # custom operators
    
    if isfile("$(@__DIR__)/../deps/deps.jl")
        include("$(@__DIR__)/../deps/deps.jl")
    else
        error("ADCME is not properly built; run `Pkg.build(\"ADCME\")` to fix the problem.")
    end
    run_metadata = nothing
    STORAGE = Dict{String, Any}()
        
    function __init__()
        global AUTO_REUSE, GLOBAL_VARIABLES, TRAINABLE_VARIABLES, UPDATE_OPS, DTYPE, COLIB
        
        copy!(tf, pyimport("tensorflow"))
        copy!(tfp, pyimport("tensorflow_probability"))

        DTYPE = Dict(Float64=>tf.float64,
            Float32=>tf.float32,
            Int64=>tf.int64,
            Int32=>tf.int32,
            Bool=>tf.bool,
            ComplexF64=>tf.complex128,
            ComplexF32=>tf.complex64,
            String=>tf.string,
            Char=>tf.string)
        AUTO_REUSE = tf.compat.v1.AUTO_REUSE
        GLOBAL_VARIABLES = tf.compat.v1.GraphKeys.GLOBAL_VARIABLES
        TRAINABLE_VARIABLES = tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES
        UPDATE_OPS = tf.compat.v1.GraphKeys.UPDATE_OPS
        global options = Options()
        colibs = include("$(@__DIR__)/../deps/CustomOps/default_formulas.jl")
        for c in colibs
            push!(COLIB, c)
        end
        
        if isfile("$(@__DIR__)/../deps/CustomOps/formulas.jl")
            colibs = include("$(@__DIR__)/../deps/CustomOps/formulas.jl")
            for c in colibs
                push!(COLIB, c)
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
    include("flow.jl")
    include("options.jl")
end

