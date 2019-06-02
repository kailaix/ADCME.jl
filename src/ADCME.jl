
__precompile__(true)
#printstyled(
#"""
#Note that `ADCME.jl` is currently a ***PROPRIETARY***  package. 
#Unauthorized copying and/or distribution without the express permission from the developers is strictly prohibited.
#""", bold=true);

module ADCME

    export tf,
    AUTO_REUSE,
    GLOBAL_VARIABLES,
    TRAINABLE_VARIABLES,
    UPDATE_OPS
    
    using PyCall

    tf = PyNULL()
    # tfp = PyNULL()
    gradients_impl = PyNULL()
    DTYPE = Dict{Type, PyObject}()
    global AUTO_REUSE, GLOBAL_VARIABLES, TRAINABLE_VARIABLES, UPDATE_OPS
    function __init__()
        global AUTO_REUSE, GLOBAL_VARIABLES, TRAINABLE_VARIABLES, UPDATE_OPS
        copy!(tf, pyimport("tensorflow"))
        copy!(gradients_impl, pyimport("tensorflow.python.ops.gradients_impl"))
        copy!(DTYPE, Dict(Float64=>tf.float64,
            Float32=>tf.float32,
            Int64=>tf.int64,
            Int32=>tf.int32,
            Bool=>tf.bool,
            ComplexF64=>tf.complex128,
            ComplexF32=>tf.complex64))
        AUTO_REUSE = tf.AUTO_REUSE
        GLOBAL_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
        TRAINABLE_VARIABLES = tf.GraphKeys.TRAINABLE_VARIABLES
        UPDATE_OPS = tf.GraphKeys.UPDATE_OPS
    end
    
    # const tf = pyimport("tensorflow")
    # const tfp = pyimport("tensorflow_probability")
    # const gradients_impl = pyimport("tensorflow.python.ops.gradients_impl")

    include("core.jl")
    include("io.jl")
    include("optim.jl")
    include("run.jl")
    include("variable.jl")
    include("ops.jl")
    include("layers.jl")
    include("sparse.jl")
    include("datasets.jl")
    include("extra.jl")
end
