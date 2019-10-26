using Random
export customop,
xavier_init,
load_op_and_grad,
load_op,
compile_op,
test_custom_op,
enable_gpu,
use_gpu

"""
    xavier_init(size, dtype=Float64)

Returns a matrix of size `size` and its values are from Xavier initialization. 
"""
function xavier_init(size, dtype=Float64)
    in_dim = size[1]
    xavier_stddev = 1. / sqrt(in_dim / 2.)
    return randn(dtype, size...)*xavier_stddev
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


############### custom operators ##################
function cmake(DIR::String="..")
    if Sys.islinux()
        run(`$CMAKE -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX $DIR`)
    else
        run(`$CMAKE $DIR`)
    end
end

load_op_dict = Dict{Tuple{String, String}, PyObject}()
load_op_grad_dict = Dict{Tuple{String, String}, PyObject}()


"""
    compile_op(oplibpath::String; check::Bool=false)

Compile the library operator by force.
"""
function compile_op(oplibpath::String; check::Bool=false)
    PWD = pwd()
    if splitext(oplibpath)[2]==""
        oplibpath = abspath(oplibpath * (Sys.islinux() ? 
                        ".so" : Sys.isapple() ? ".dylib" : ".dll"))
    end
    if check && isfile(oplibpath)
        return 
    end
    DIR, FILE = splitdir(oplibpath)
    if !isdir(DIR); mkdir(DIR); end 
    cd(DIR)
    try
        cmake()
        run(`$MAKE -j`)
    catch
        @warn("Compiling not successful. Instruction: Check $oplibpath")
    finally
        cd(PWD)
    end
end

@doc """
    load_op(oplibpath::String, opname::String)

Loads the operator `opname` from library `oplibpath`.
"""
function load_op(oplibpath::String, opname::String)
    if splitext(oplibpath)[2]==""
        oplibpath = abspath(oplibpath * (Sys.islinux() ? 
                        ".so" : Sys.isapple() ? ".dylib" : ".dll"))
    end
    oplibpath = abspath(oplibpath)
    if haskey(load_op_dict, (oplibpath,opname))
        return load_op_dict[(oplibpath,opname)]
    end

    if !isfile(oplibpath)
        error("File $oplibpath does not exist. Instruction:\nRunning `compile_op(oplibpath)` to compile the library first.")
    end
    fn_name = opname*randstring(8)
py"""
import tensorflow as tf
lib$$fn_name = tf.load_op_library($oplibpath)
"""
    lib = py"lib$$fn_name"
    s = getproperty(lib, opname)
    load_op_dict[(oplibpath,opname)] = s
    printstyled("Load library operator: $oplibpath ==> $opname\n", color=:green)
    return s
end

@doc """
    load_op_and_grad(oplibpath::String, opname::String; multiple::Bool=false)

Loads the operator `opname` from library `oplibpath`; gradients are also imported. 
If `multiple` is true, the operator is assumed to have multiple outputs. 
"""
function load_op_and_grad(oplibpath::String, opname::String; multiple::Bool=false)
    if splitext(oplibpath)[2]==""
        oplibpath = oplibpath * (Sys.islinux() ? 
                        ".so" : Sys.isapple() ? ".dylib" : ".dll")
    end
    oplibpath = abspath(oplibpath)
    if haskey(load_op_grad_dict, (oplibpath,opname))
        return load_op_grad_dict[(oplibpath,opname)]
    end
    if !isfile(oplibpath)
        error("File $oplibpath does not exist. Instruction:\nRunning `compile_op(oplibpath)` to compile the library first.")
    end
    
    opname_grad = opname*"_grad"
    fn_name = opname*randstring(8)
if !multiple
py"""
import tensorflow as tf
lib$$fn_name = tf.load_op_library($oplibpath)
@tf.custom_gradient
def $$fn_name(*args):
    u = lib$$fn_name.$$opname(*args)
    def grad(dy):
        return lib$$fn_name.$$opname_grad(dy, u, *args)
    return u, grad
"""
else
py"""
import tensorflow as tf
lib$$fn_name = tf.load_op_library($oplibpath)
@tf.custom_gradient
def $$fn_name(*args):
    u = lib$$fn_name.$$opname(*args)
    def grad(*dy):
        dy = [y for y in dy if y is not None]
        return lib$$fn_name.$$opname_grad(*dy, *u, *args)
    return u, grad
"""
end
        s = py"$$fn_name"
        load_op_grad_dict[(oplibpath,opname)] = s
        printstyled("Load library operator (with gradient): $oplibpath ==> $opname\n", color=:green)
        return s
end

"""
    load_system_op(s::String, oplib::String, grad::Bool=true)

Loads custom operator from CustomOps directory (shipped with ADCME instead of TensorFlow)
For example 
```
s = "SparseOperator"
oplib = "libSO"
grad = true
```
this will direct Julia to find library `CustomOps/SparseOperator/libSO.dylib` on MACOSX
"""
function load_system_op(s::String, oplib::String, opname::String, grad::Bool=true; return_str::Bool=false)
    dir = joinpath(joinpath("$(@__DIR__)", "../deps/CustomOps"), s)
    if !isdir(dir)
        error("Folder for the operator $s does not exist: $dir")
    end
    oplibpath = joinpath(joinpath(dir, "build"), oplib)
    # check if the library exists 
    libfile = oplibpath * (Sys.islinux() ? 
                        ".so" : Sys.isapple() ? ".dylib" : ".dll")
    # @show libfile
    if !isfile(libfile)
        @info "Lib $s exists in registery but was not initialized. Compiling..."
        compile(s)
    end
    if return_str
        return oplibpath
    end
    if grad
        load_op_and_grad(oplibpath, opname)
    else
        load_op(oplibpath, opname)
    end
end

"""
    compile(s::String)

Compiles the library `s` by force.
"""
function compile(s::String)
    PWD = pwd()
    dir = joinpath(joinpath("$(@__DIR__)", "../deps/CustomOps"), s)
    if !isdir(dir)
        error("Folder for the operator $s does not exist: $dir")
    end
    cd(dir)
    rm("build",force=true,recursive=true)
    mkdir("build")
    cd("build")
    try
        cmake()
        run(`$MAKE -j`)
    catch e 
        error("Compilation error: $e")
    finally
        cd(PWD)
    end
end

"""
    customop()

Create a new custom operator.

# example
```julia-repl
julia> customop() # create an editable `customop.txt` file
[ Info: Edit custom_op.txt for custom operators
julia> customop() # after editing `customop.txt`, call it again to generate interface files.
```
"""
function customop()
    # install_custom_op_dependency()
    py_dir = "$(@__DIR__)/../examples/custom_op/template"
    if !("custom_op.txt" in readdir("."))
        cp("$(py_dir)/custom_op.example", "custom_op.txt")
        @info "Edit custom_op.txt for custom operators"
        return
    else
        python = PyCall.python
        run(`$python $(py_dir)/customop.py custom_op.txt $py_dir`)
    end
end


function test_custom_op()
    PWD = pwd()
    cd("$(@__DIR__)/../deps/CustomOps/SparseSolver")
    rm("build", recursive=true, force=true)
    mkdir("build")
    cd("build")
    cmake()
    run(`$MAKE -j`)
    include("$(@__DIR__)/../deps/CustomOps/SparseSolver/gradtest.jl")
    cd(PWD)
    true
end

function enable_gpu()
    pkgs = Conda._installed_packages()

    if !("tensorflow-gpu" in pkgs)
        Conda.add("tensorflow-gpu=1.14")
    end
    
    if !("cudatoolkit" in pkgs)
        Conda.add("cudatoolkit", channel="anaconda")
    end

    gpus = joinpath(splitdir(tf.__file__)[1], "include/third_party/gpus")
    if !isdir(gpus)
    mkdir(gpus)
    end
    gpus = joinpath(gpus, "cuda")
    if !isdir(gpus)
    mkdir(gpus)
    end
    incpath = joinpath(splitdir(strip(read(`which nvcc`, String)))[1], "../include/")
    if !isdir(joinpath(gpus, "include"))
        cp(incpath, joinpath(gpus, "include"))
    end

    pth = joinpath(Conda.ROOTENV, "pkgs/cudatoolkit-10.1.168-0/lib/")
    # compatible 
    files = readdir(pth)
    for f in files
        if f[end-2:end]==".10" && !isfile(joinpath(pth, f*".0"))
            symlink(joinpath(pth, f), joinpath(pth, f*".0"))
        end
        if f[end-4:end]==".10.1" && !isfile(joinpath(pth, f[1:end-2]*".0"))
            symlink(joinpath(pth, f), joinpath(pth, f[1:end-2]*".0"))
        end
    end
    
    println("Run the following command in shell

echo 'export LD_LIBRARY_PATH=$pth:\$LD_LIBRARY_PATH' >> ~/.bashrc")
end

function use_gpu(i::Union{Nothing,Int64}=nothing)
    dl = pyimport("tensorflow.python.client.device_lib")
    if !isnothing(i) && i>=1
        i = join(collect(0:i-1),',') 
        ENV["CUDA_VISIBLE_DEVICES"] = i 
    end
    local_device_protos = dl.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]
end

