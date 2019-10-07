
__precompile__(true)
module ADCME

    export tf,
    AUTO_REUSE,
    GLOBAL_VARIABLES,
    TRAINABLE_VARIABLES,
    UPDATE_OPS
    
    using Suppressor
    using PyCall
    using Random
    using LinearAlgebra
    using PyPlot
    using Conda
    

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
        "set_tensor_flow_timer"=>("Timer", "libTensorFlowTimer", "set_tensor_flow_timer", false),
        "sparse_mat_mul"=>("SparseMatMul", "libSparseMatMul", "sparse_sparse_mat_mul", false),
        "diag_sparse_mat_mul"=>("SparseMatMul", "libSparseMatMul", "sparse_sparse_mat_mul", false),
        "sparse_diag_mat_mul"=>("SparseMatMul", "libSparseMatMul", "sparse_sparse_mat_mul", false),
        "sparse_indexing"=>("SparseIndexing", "libSparseIndexing", "sparse_indexing", false)
    )

    libSuffix = Sys.isapple() ? "dylib" : (Sys.islinux() ? "so" : "dll")
    
    CC = joinpath(Conda.BINDIR, "gcc")
    CXX = joinpath(Conda.BINDIR, "g++")
    CMAKE = joinpath(Conda.BINDIR, "cmake")
    MAKE = joinpath(Conda.BINDIR, "make")
    TFLIB = nothing
    

    function __init__()
        # install_custom_op_dependency() # always install dependencies
        global AUTO_REUSE, GLOBAL_VARIABLES, TRAINABLE_VARIABLES, UPDATE_OPS, DTYPE, TFLIB
        if haskey(ENV, "LD_LIBRARY_PATH")
            ENV["LD_LIBRARY_PATH"] = Conda.LIBDIR*":"*ENV["LD_LIBRARY_PATH"]
        else
            ENV["LD_LIBRARY_PATH"] = Conda.LIBDIR
        end
            
        PYTHON = joinpath(Conda.BINDIR, "python")
        
        if PYTHON!=PyCall.python
            error("""PyCall python and TensorFlow python does not match.
$(PyCall.python) vs $PYTHON
Rebuild PyCall with 

julia> ENV["PYTHON"] = "$PYTHON"
julia> using Pkg; Pkg.build("PyCall")

""")
        end
        
        copy!(tf, (@suppress pyimport_conda("tensorflow","tensorflow")))
        copy!(tfops, (@suppress pyimport_conda("tensorflow.python.framework.ops","tensorflow")))
        copy!(tfp, (@suppress pyimport_conda("tensorflow_probability","tensorflow_probability")))
        copy!(gradients_impl, (@suppress pyimport_conda("tensorflow.python.ops.gradients_impl","tensorflow")))
        copy!(pickle, (@suppress pyimport_conda("pickle", "pickle")))
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
        TFLIB = joinpath(splitdir(tf.__file__)[1], "libtensorflow_framework.so")
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
