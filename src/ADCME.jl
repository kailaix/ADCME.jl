
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
    gradients_impl = PyNULL()
    DTYPE = Dict{Type, PyObject}()
    # a list of custom operators 
    COLIB = Dict{String, Tuple{String, String, String, Bool}}(
        "sparse_solver"=>("SparseSolver", "libSparseSolver", "sparse_solver", true),
        "sparse_assembler"=>("SparseAccumulate", "libSparseAccumulate", "", false),
        "sparse_least_square"=>("SparseLeastSquare", "libSparseLeastSquare", "sparse_least_square", true)
    )

    libSuffix = Sys.isapple() ? "dylib" : (Sys.islinux() ? "so" : "dll")

    function install_tensorflow()
        if haskey(ENV,"REINSTALL_PIP")
            @info "Reinstall pip..."
            download("https://bootstrap.pypa.io/get-pip.py", "get-pip.py")
            run(`$(PyCall.python) get-pip.py --user`)
            rm("get-pip.py")
        end
        try 
            run(`$(PyCall.python) -m pip --version`)
        catch
            @warn "pip is not installed, downloading and installing pip..."
            download("https://bootstrap.pypa.io/get-pip.py", "get-pip.py")
            run(`$(PyCall.python) get-pip.py --user`)
            rm("get-pip.py")
        end
        run(`$(PyCall.python) -m pip install --user -U numpy`)
        run(`$(PyCall.python) -m pip install --user tensorflow==1.14`)
        run(`$(PyCall.python) -m pip install --user tensorflow_probability==0.7`)
    end
    
    
    function __init__()
        global AUTO_REUSE, GLOBAL_VARIABLES, TRAINABLE_VARIABLES, UPDATE_OPS, DTYPE
        if haskey(ENV,"REINSTALL_PIP")
            install_tensorflow()
            PYTHONPATH="/home/travis/.local/lib/python3.5/site-packages"
        end
        
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
