
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

    tf = PyNULL()
    tfp = PyNULL()
    tfops = PyNULL()
    pickle = PyNULL()
    gradients_impl = PyNULL()
    DTYPE = Dict{Type, PyObject}()
    # a list of custom operators 
    COLIB = Dict{String, Tuple{String, String, String, Bool}}(
        "sparse_solver"=>("SparseSolver", "libSparseSolver", "sparse_solver", true),
        "sparse_assembler"=>("SparseAccumulate", "libSparseAccumulate", "", false),
        "sparse_least_square"=>("SparseLeastSquare", "libSparseLeastSquare", "sparse_least_square", true),
        "get_tensor_flow_timer"=>("Timer", "libTensorFlowTimer", "get_tensor_flow_timer", false),
        "set_tensor_flow_timer"=>("Timer", "libTensorFlowTimer", "set_tensor_flow_timer", false)
    )

    libSuffix = Sys.isapple() ? "dylib" : (Sys.islinux() ? "so" : "dll")
    
    
    function __init__()
        global AUTO_REUSE, GLOBAL_VARIABLES, TRAINABLE_VARIABLES, UPDATE_OPS, DTYPE
        copy!(tf, pyimport("tensorflow"))
        copy!(tfops, pyimport("tensorflow.python.framework.ops"))
        copy!(tfp, pyimport("tensorflow_probability"))
        copy!(gradients_impl, pyimport("tensorflow.python.ops.gradients_impl"))
        copy!(pickle, pyimport("pickle"))
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
    include("gan.jl")
end
