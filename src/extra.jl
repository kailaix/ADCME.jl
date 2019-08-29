using Random
export customop,
torchexample,
xavier_init,
gan,
klgan,
wgan,
rklgan,
lsgan,
test_customop,
load_op_and_grad,
load_op


"""
    customop(torch=false; julia=false)

Create a new custom operator.

# example
```julia-repl
julia> customop() # create an editable `customop.txt` file
[ Info: Custom operator wrapper generated; Torch is disabled

julia> customop() # after editing `customop.txt`, call it again to generate interface files.
[ Info: Custom operator wrapper generated; Torch is disabled
```
The option `torch` adds support for `PyTorch` backend in `CMakeLists.txt`
"""
function customop(torch=false; julia=false)
    py_dir = "$(@__DIR__)/../examples/custom_op/template"
    if !("custom_op.txt" in readdir("."))
        cp("$(py_dir)/custom_op.example", "custom_op.txt")
        @info "Edit custom_op.txt for custom operators"
        return
    else
        python = PyCall.python
        run(`$python $(py_dir)/customop.py custom_op.txt $py_dir $(torch ? "" : "# ")`)
        @info "Custom operator wrapper generated; Torch is $(torch ? "enabled" : "disabled")"
    end

    if julia
        cmakelist = read("CMakeLists.txt", String)
        s1 = """set(CMAKE_CXX_FLAGS "-std=c++11 \${CMAKE_CXX_FLAGS}")"""
        s2 = """execute_process(COMMAND python -c "from sysconfig import get_paths as gp; import sys; sys.stdout.write(gp()['include'])" OUTPUT_VARIABLE PYTHON_INC)
execute_process(COMMAND julia -e "using PyCall; print(PyCall.libpython)" OUTPUT_VARIABLE PYTHON_LIB)
execute_process(COMMAND julia -e "abspath(joinpath(Sys.BINDIR, \\"../lib\\"))|>print" OUTPUT_VARIABLE JULIA_LIB)
execute_process(COMMAND julia -e "abspath(joinpath(Sys.BINDIR, \\"../include/julia\\"))|>print" OUTPUT_VARIABLE JULIA_INC)
add_definitions( -DJULIA_ENABLE_THREADING=1 )
"""
        cmakelist = replace(cmakelist, s1=>s2)
        s1 = """link_directories(\${TF_LIB})"""
        s2 = """include_directories(\${JULIA_INC} \${PYTHON_INC})
link_directories(\${TF_LIB} \${JULIA_LIB})"""
        cmakelist = replace(cmakelist, s1=>s2)
        s1 = """tensorflow_framework"""
        s2 = """tensorflow_framework julia \${PYTHON_LIB}"""
        cmakelist = replace(cmakelist, s1=>s2)
        write("CMakeLists.txt", cmakelist)

        gradtest = read("gradtest.jl", String)
        m = match(r"(\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\# Load Operator.*End Load Operator \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#)"s, gradtest)
        s = m.captures[1]
        m = match(r"(build/.*?)\.so", s)
        Dir = m.captures[1]
        m = match(r"= py\\\"(.*?)\\\"", s)
        Fun = m.captures[1]
        s0 = "$Fun = load_op(\"$Dir\", \"$Fun\")"
        gradtest = replace(gradtest, s=>s0)
        m = match(r"(\# TODO: change your test parameter to `m`.*)"s, gradtest)
        s = m.captures[1]
        gradtest = replace(gradtest, s=>"")
        write("gradtest.jl", gradtest)

        opname = strip(readline("custom_op.txt"))
        cpp = read("$opname.cpp", String)
        s = "(REGISTER_OP\\(\"$(opname)Grad\"\\).*)"
        r = Regex(s, "s")
        m = match(r, cpp)
        cpp = replace(cpp, m.captures[1]=>"")
        cpp = replace(cpp, "using namespace tensorflow;"=>"using namespace tensorflow;
#include \"julia.h\"\n#include \"Python.h\"")
        write("$opname.cpp", cpp)

        cp("$(py_dir)/julia_op_example.h", "example.h")
        
    end

end

function torchexample()
    filename = "$(@__DIR__)/../examples/torch/laexample.cpp"
    s = read(filename, String)
    println(s)
end

function xavier_init(size, dtype=Float64)
    in_dim = size[1]
    xavier_stddev = 1. / sqrt(in_dim / 2.)
    return randn(dtype, size...)*xavier_stddev
end

"""
D_loss, G_loss = klgan(P::PyObject, Q::PyObject, discriminator::Function)

return discriminator loss and generator loss for KL divergence
`P` is the real distribution, `Q` is the generated distribution, 
`discriminator` is a critic function that outputs values in (0,1) (e.g. the last activation function is sigmoid)
"""
function klgan(P::PyObject, Q::PyObject, discriminator::Function)
    D_real = discriminator(P)
    D_fake = discriminator(Q)
    D_loss = -mean(log(D_real) + log(1-D_fake))
    G_loss = mean(log((1-D_fake)/D_fake))
    D_loss, G_loss
end

"""
D_loss, G_loss = gan(P::PyObject, Q::PyObject, discriminator::Function)

return discriminator loss and generator loss for JS divergence
`P` is the real distribution, `Q` is the generated distribution, 
`discriminator` is a critic function that outputs values in (0,1) (e.g. the last activation function is sigmoid)
"""
function gan(P::PyObject, Q::PyObject, discriminator::Function)
    D_real = discriminator(P)
    D_fake = discriminator(Q)
    D_loss = -mean(log(D_real) + log(1-D_fake))
    G_loss = -mean(log(D_fake))
    D_loss, G_loss
end

"""
D_loss, G_loss = wgan(P::PyObject, Q::PyObject, discriminator::Function)

return discriminator loss and generator loss for 1 Wasserstein
`P` is the real distribution, `Q` is the generated distribution, 
No constraint is imposed on discriminator
`clamp` is required for the discriminator weights
"""
function wgan(P::PyObject, Q::PyObject, discriminator::Function)
    D_real = discriminator(P)
    D_fake = discriminator(Q)
    D_loss = mean(D_fake)-mean(D_real)
    G_loss = -mean(D_fake)
    D_loss, G_loss
end

"""
D_loss, G_loss = rklgan(P::PyObject, Q::PyObject, discriminator::Function)

return discriminator loss and generator loss for reverse KL divergence
`P` is the real distribution, `Q` is the generated distribution, 
`discriminator` is a critic function that outputs values in (0,1) (e.g. the last activation function is sigmoid)
"""
function rklgan(P::PyObject, Q::PyObject, discriminator::Function)
    D_real = discriminator(P)
    D_fake = discriminator(Q)
    G_loss = mean(log((1-D_fake)/D_fake))
    D_loss = -mean(log(D_fake)+log(1-D_real))
    D_loss, G_loss
end


"""
D_loss, G_loss = lsgan(P::PyObject, Q::PyObject, discriminator::Function)

return discriminator loss and generator loss for least square
`P` is the real distribution, `Q` is the generated distribution, 
`discriminator` is a critic function that outputs values in (0,1) (e.g. the last activation function is sigmoid)
1 for real, 0 for fake
"""
function lsgan(P::PyObject, Q::PyObject, discriminator::Function)
    D_real = discriminator(P)
    D_fake = discriminator(Q)
    D_loss = mean((D_real-1)^2+D_fake^2)
    G_loss = mean((D_fake-1)^2)
    D_loss, G_loss
end

export traintestdev
function traintestdev(n::Int64, train::Float64=0.64, test::Float64=0.2)
    rn = randperm(n)
    if train+test>1 || train<0 || test<0
        error("invalid train and test set size")
    end
    dev = 1 - train-test
    return rn[1:Int64(round(train*n))], rn[Int64(round(train*n))+1:Int64(round((train+test)*n))],
                rn[Int64(round((train+test)*n)):end]
end


function test_customop()
    dir = pwd()
    @info "Test: $(@__DIR__)/../examples/while_loop/DirichletBD"
    cd("$(@__DIR__)/../examples/while_loop/DirichletBD")
    if !isdir("build")
        mkdir("build")
    end
    cd("build")
    run(`cmake ..`)
    run(`make -j`)
    cd("..")
    include("gradtest.jl")
    

    @info "Test: $(@__DIR__)/../examples/while_loop/SparseSolver"
    cd("$(@__DIR__)/../examples/while_loop/SparseSolver")
    if !isdir("build")
        mkdir("build")
    end
    cd("build")
    run(`cmake ..`)
    run(`make -j`)
    cd("..")
    include("gradtest.jl")

    cd(dir)
end 


@doc """
load_op(oplibpath::String, opname::String; grad=false)

loads the operator `opname` from library `oplibpath`, 
if the surfix of `oplibpath` is not given, it will be inferred from system
"""
function load_op(oplibpath::String, opname::String; grad=false)
    if splitext(oplibpath)[2]==""
        oplibpath = oplibpath * (Sys.islinux() ? 
                        ".so" : Sys.isapple() ? ".dylib" : ".dll")
    end
try
py"""
import tensorflow as tf
libop = tf.load_op_library($oplibpath)
"""
    lib = py"libop"
    s = getproperty(lib, opname)
    println("Load library: $oplibpath")
    if grad
        t = getproperty(lib, opname*"_grad")
        return s, t
    else
        return s
    end
catch e
    @warn("Library $oplibpath was not loaded:\n$e")
end
end

function load_op_and_grad(oplibpath::String, opname::String)
    if splitext(oplibpath)[2]==""
        oplibpath = oplibpath * (Sys.islinux() ? 
                        ".so" : Sys.isapple() ? ".dylib" : ".dll")
    end
    opname_grad = opname*"_grad"
try
py"""
import tensorflow as tf
lib = tf.load_op_library($oplibpath)
@tf.custom_gradient
def sparse_solver(*args):
    u = lib.__getattribute__($opname)(*args)
    def grad(dy):
        return lib.__getattribute__($opname_grad)(dy, u, *args)
    return u, grad
"""
    s = py"sparse_solver"
    println("Load library: $oplibpath")
    return s
catch e
    @warn("Library $oplibpath was not loaded:\n$e")
end
end