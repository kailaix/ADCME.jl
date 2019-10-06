using Random
export customop,
torchexample,
xavier_init,
load_op_and_grad,
load_op,
compile_op,
tic,
toc,
test_custom_op

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
function cmake()
    if Sys.islinux()
        run(`$CMAKE -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX ..`)
    else
        run(`$CMAKE ..`)
    end
end

load_op_dict = Dict{Tuple{String, String}, PyObject}()
load_op_grad_dict = Dict{Tuple{String, String}, PyObject}()


"""
    compile_op(oplibpath::String, opname::String)

Compile the library operator by force.
"""
function compile_op(oplibpath::String)
    PWD = pwd()
    if splitext(oplibpath)[2]==""
        oplibpath = abspath(oplibpath * (Sys.islinux() ? 
                        ".so" : Sys.isapple() ? ".dylib" : ".dll"))
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

loads the operator `opname` from library `oplibpath`, 
if the surfix of `oplibpath` is not given, it will be inferred from system
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
function load_system_op(s::String, oplib::String, grad::Bool=true)

Load custom operator from CustomOps directory (shipped with ADCME instead of TensorFlow)
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
Compile the library `s` by force.
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
    # install_custom_op_dependency()
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
        s1 = """{TF_LIB_FILE}"""
        s2 = """{TF_LIB_FILE} julia \${PYTHON_LIB}"""
        cmakelist = replace(cmakelist, s1=>s2)
        write("CMakeLists.txt", cmakelist)

        gradtest = read("gradtest.jl", String)
        gradtest = replace(gradtest, "load_op_and_grad"=>"load_op")
        gradtest = replace(gradtest, ", multiple=true"=>"")
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



"""
    tic(o::PyObject, i::Union{PyObject, Integer}=0)

Construts a TensorFlow timer with index `i`. The start time record is right before `o` is executed.
"""
function tic(o::PyObject, i::Union{PyObject, Integer}=0)
    set_tensor_flow_timer = load_system_op(COLIB["set_tensor_flow_timer"]...)
    i = convert_to_tensor(i, dtype=Int32)
    start_timer = set_tensor_flow_timer(i)
    o = bind(o, start_timer)
end


"""
    toc(o::PyObject, i::Union{PyObject, Integer}=0)

Returns the elapsed time from last [`tic`](@ref) call with index `i` (default=0). The terminal time record is right before `o` is executed.
"""
function toc(o::PyObject, i::Union{PyObject, Integer}=0)
    get_tensor_flow_timer = load_system_op(COLIB["get_tensor_flow_timer"]...)
    i = convert_to_tensor(i, dtype=Int32)
    end_timer = get_tensor_flow_timer(i)
    o = bind(o, end_timer)
    o, end_timer
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