
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
    libadcme = missing

    #------------------------------------------------------------------------------------------
    # Global Storage 
    DTYPE = Dict{Type, PyObject}()
    LIBADCME = abspath(joinpath("$(@__DIR__)", "../deps/CustomOps/build", "libadcme.$dlext"))
    if Sys.iswindows()
        LIBADCME = abspath(joinpath("$(@__DIR__)", "../deps/CustomOps/build", "adcme.dll"))
    end
    LIBPLUGIN = abspath(joinpath("$(@__DIR__)", "../deps/Plugin"))
    
    if isfile("$(@__DIR__)/../deps/deps.jl")
        include("$(@__DIR__)/../deps/deps.jl")
        if Sys.iswindows()
            ENV["PATH"] = LIBDIR*";"*ENV["PATH"]
        else
            if haskey(ENV, "LD_LIBRARY_PATH")
                ENV["LD_LIBRARY_PATH"] = LIBDIR*":"*ENV["LD_LIBRARY_PATH"]
            else
                ENV["LD_LIBRARY_PATH"] = LIBDIR
            end
        end
    else
        error("ADCME is not properly built; run `Pkg.build(\"ADCME\")` to fix the problem.")
    end
    run_metadata = nothing
    STORAGE = Dict{String, Any}()
        
    function __init__()
        global AUTO_REUSE, GLOBAL_VARIABLES, TRAINABLE_VARIABLES, UPDATE_OPS, DTYPE, libadcme
        copy!(tf, pyimport("tensorflow"))
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        try
            copy!(tfp, pyimport("tensorflow_probability"))
        catch
        end

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

        try 
            libadcme = tf.load_op_library(LIBADCME)
        catch 
            @warn "Cannot load $LIBADCME. Please recompile the shared library by `ADCME.precompile()` for using custom operators."
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
    include("install.jl")
    include("sparse.jl")
    include("random.jl")
    include("gan.jl")
    include("ot.jl")
    include("ode.jl")
    include("flow.jl")
    include("options.jl")
    include("mpi.jl")
    include("toolchain.jl")
    include("kit.jl")
    include("rbf.jl")
    include("pcl.jl")
    include("sqlite.jl")
end

