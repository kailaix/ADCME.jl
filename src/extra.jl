using Random
export customop,
xavier_init,
load_op_and_grad,
load_op,
use_gpu,
test_jacobian,
install,
load_system_op,
install_adept,
register,
debug,
doctor,
nnuq,
compile,
list_physical_devices,
MCMCSimple,
simulate,
diagnose

"""
    xavier_init(size, dtype=Float64)

Returns a matrix of size `size` and its values are from Xavier initialization. 
"""
function xavier_init(size, dtype=Float64)
    in_dim = size[1]
    xavier_stddev = 1. / sqrt(in_dim / 2.)
    return randn(dtype, size...)*xavier_stddev
end

############### custom operators ##################
function cmake(DIR::String=".."; CMAKE_ARGS::String = "")
    ENV_ = copy(ENV)
    LD_PATH = Sys.iswindows() ? "PATH" : "LD_LIBRARY_PATH"
    if haskey(ENV_, LD_PATH)
        ENV_[LD_PATH] = ENV[LD_PATH]*":$LIBDIR"
    else
        ENV_[LD_PATH] = LIBDIR
    end
    if Sys.iswindows()
        if !haskey(ENV_, "VS150COMNTOOLS")
            # @warn "VS150COMNTOOLS is not set, default to /c/Program Files (x86)/Microsoft Visual Studio/2017/Community/Common7/Tools" maxlog=1
            ENV_["VS150COMNTOOLS"] = "/c/Program Files (x86)/Microsoft Visual Studio/2017/Community/Common7/Tools"
        end
        # @info "Do remember to add ADD_DEFINITIONS(-DNOMINMAX) to your CMakeLists.txt" maxlog=1
        run(setenv(`$CMAKE -G"Visual Studio 15" -DJULIA="$(joinpath(Sys.BINDIR, "julia"))" -A x64 $CMAKE_ARGS $DIR`, ENV_)) # very important, x64
    else
        run(setenv(`$CMAKE -DJULIA="$(joinpath(Sys.BINDIR, "julia"))" -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX $CMAKE_ARGS $DIR`, ENV_))
    end
end

function make()
    ENV_ = copy(ENV)
    LD_PATH = Sys.iswindows() ? "PATH" : "LD_LIBRARY_PATH"
    if haskey(ENV_, LD_PATH)
        ENV_[LD_PATH] = ENV[LD_PATH]*":$LIBDIR"
    else
        ENV_[LD_PATH] = LIBDIR
    end
    if Sys.iswindows()
        sln_file = filter(x->endswith(x, ".sln"), readdir())
        if length(sln_file)==0
            error("No .sln file found. Did you run `ADCME.cmake()`?")
        elseif length(sln_file)>1
            error("More than 1 .sln file found. Check your program.")
        else
            sln_file = sln_file[1]
        end 
        run(`cmd /c $CMAKE  --build . -j --target ALL_BUILD --config Release`)
    else
        run(setenv(`$MAKE -j $(Sys.CPU_THREADS)`, ENV_))
    end
end

"""
    make_library(Libdir::String)

Make shared library in `Libdir`. The structure of the source codes files are 

```
- Libdir 
  - *.cpp 
  - *.h 
  - CMakeLists
  - build (Optional)
```
"""
function make_library(Libdir::String)
   if !isdir(Libdir)
    error("$Libdir is not a valid directory.")
   end
   PWD = pwd()
   cd(Libdir)
   if !isdir("build")
    mkdir("build")
   end
   cd("build")
   if !isfile("Makefile")
    ADCME.cmake()
   end
   ADCME.make()
   cd(PWD)
end

load_op_dict = Dict{Tuple{String, String}, PyObject}()
load_op_grad_dict = Dict{Tuple{String, String}, PyObject}()


@doc """
    load_op(oplibpath::String, opname::String)

Loads the operator `opname` from library `oplibpath`.
"""
function load_op(oplibpath::String, opname::String; verbose::Bool = true)
    if Sys.iswindows()
        a, b = splitdir(oplibpath)
        if length(b)>=3 && b[1:3]=="lib"
            b = b[4:end]
        end
        oplibpath = joinpath(a, b)
    end
    if splitext(oplibpath)[2]==""
        oplibpath = abspath(oplibpath * (Sys.islinux() ? 
                        ".so" : Sys.isapple() ? ".dylib" : ".dll"))
    end
    oplibpath = abspath(oplibpath)
    if haskey(load_op_dict, (oplibpath,opname))
        return load_op_dict[(oplibpath,opname)]
    end

    if !isfile(oplibpath)
        error("File $oplibpath does not exist. Instruction:\nRunning `compile(oplibpath)` to compile the library first.")
    end
    fn_name = opname*randstring(8)
try 
py"""
import tensorflow as tf
lib$$fn_name = tf.load_op_library($oplibpath)
"""
catch(e)
    printstyled("Failed to open $oplibpath. Error Message from the TensorFlow backend\n$(string(e))\n", color=:red)
    Libdl.dlopen(oplibpath)
end
    lib = py"lib$$fn_name"
    s = getproperty(lib, opname)
    load_op_dict[(oplibpath,opname)] = s
    verbose && printstyled("Load library operator: $oplibpath ==> $opname\n", color=:green)
    return s
end

@doc """
    load_op_and_grad(oplibpath::String, opname::String; multiple::Bool=false)

Loads the operator `opname` from library `oplibpath`; gradients are also imported. 
If `multiple` is true, the operator is assumed to have multiple outputs. 
"""
function load_op_and_grad(oplibpath::String, opname::String; multiple::Bool=false, verbose::Bool = true)
    if Sys.iswindows()
        a, b = splitdir(oplibpath)
        if length(b) >=3 && b[1:3]=="lib"
            b = b[4:end]
        end
        oplibpath = joinpath(a, b)
    end
    if splitext(oplibpath)[2]==""
        oplibpath = oplibpath * (Sys.islinux() ? 
                        ".so" : Sys.isapple() ? ".dylib" : ".dll")
    end
    oplibpath = abspath(oplibpath)
    if haskey(load_op_grad_dict, (oplibpath,opname))
        return load_op_grad_dict[(oplibpath,opname)]
    end
    if !isfile(oplibpath)
        error("File $oplibpath does not exist. Instruction:\nRunning `compile(oplibpath)` to compile the library first.")
    end
    
    opname_grad = opname*"_grad"
    fn_name = opname*randstring(8)
    try
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
        dy = [y for y in dy if y is not None and y.dtype in [tf.float64, tf.float32]] # only float64 and float32 can backpropagate gradients
        return lib$$fn_name.$$opname_grad(*dy, *u, *args)
    return u, grad
"""
end
catch(e)
    printstyled("Failed to open $oplibpath. Error Message from the TensorFlow backend\n$(string(e))\n", color=:red)
    Libdl.dlopen(oplibpath)
end
        s = py"$$fn_name"
        load_op_grad_dict[(oplibpath,opname)] = s
        verbose && printstyled("Load library operator (with gradient, multiple outputs = $multiple): $oplibpath ==> $opname\n", color=:green)
        return s
end


"""
    load_system_op(opname::String, grad::Bool=true; multiple::Bool=false)

Loads custom operator from CustomOps directory (shipped with ADCME instead of TensorFlow)
For example 
```
s = "SparseOperator"
oplib = "libSO"
grad = true
```
this will direct Julia to find library `CustomOps/SparseOperator/libSO.dylib` on MACOSX
"""
function load_system_op(opname::String, grad::Bool=true; multiple::Bool=false)
    if !isfile(LIBADCME)
        @info "$LIBADCME does not exist. Precompiling..."
        ADCME.precompile()
    end
    if grad
        load_op_and_grad(LIBADCME, opname; multiple=multiple, verbose=false)
    else
        load_op(LIBADCME, opname, verbose=false)
    end
end

"""
    compile(s::String; force::Bool=false)

Compiles the library given by path `deps/s`. If `force` is false, `compile` first check whether 
the binary product exists. If the binary product exists, return 2. Otherwise, `compile` tries to 
compile the binary product, and returns 0 if successful; it return 1 otherwise. 
"""
function compile(s::String; force::Bool=false, customdir::Bool = false)
    PWD = pwd()
    dir = s 
    if !customdir
        dir = joinpath(joinpath("$(@__DIR__)", "../deps/CustomOps"), s)
    end
    if !isdir(dir)
        @warn("Folder for the operator $s does not exist: $dir")
        return 1
    end
    cd(dir)
    
    local surfix 
    if Sys.isapple()
        surfix = ".dylib"
    elseif Sys.islinux()
        surfix = ".so"
    elseif Sys.iswindows()
        surfix = ".dll"
    end
    if !force && isdir("build") # check if product exists 
        files = readdir("build")
        if any([endswith(x, surfix) for x in files])
            @warn("The binary product exists.")
            cd(PWD)
            return 2
        end
    end
    rm("build",force=true,recursive=true)
    mkdir("build")
    cd("build")
    try
        cmake()
        make()
        cd(PWD)
        return 0
    catch e 
        error("Compilation error: $e")
        cd(PWD)
        return 1
    end
end

"""
    precompile(force::Bool=true)

Compiles all the operators in `formulas.txt`. 
"""
function Base.:precompile(force::Bool=true)
    PWD = pwd()
    if (!force) && isdir("$(@__DIR__)/../deps/CustomOps/build")
        return 
    end
    @info "Compiling ADCME built-in custom operators..."
    cd("$(@__DIR__)/../deps/CustomOps")
    rm("build", force=true, recursive=true)
    mkdir("build")
    cd("build")
    ADCME.cmake()
    ADCME.make()
    cd(PWD)
end

"""
    compile()

Compile a custom operator in the current directory. A `CMakeLists.txt` must be present. 
"""
function compile()
    PWD = pwd()
    if !isfile("CMakeLists.txt")
        error(SystemError("No CMakeLists.txt in the current directory found."))
    end
    if !isdir("build")
        mkdir("build")
    end
    cd("build")
    try 
        cmake()
        make()
    catch e
        @warn "Compiling failed: $e"
    finally
        cd(PWD)
    end
end


"""
    customop()

Create a new custom operator. Typically users call `customop` twice: the first call generates a `customop.txt`, 
users edit the content in the file; the second all generates C++ source code, CMakeLists.txt, and gradtest.jl from `customop.txt`.

# Example
```julia-repl
julia> customop() # create an editable `customop.txt` file
[ Info: Edit custom_op.txt for custom operators
julia> customop() # after editing `customop.txt`, call it again to generate interface files.
```
"""
function customop()
    # install_custom_op_dependency()
    py_dir = "$(@__DIR__)/../deps/CustomOpsTemplate"
    if !("custom_op.txt" in readdir("."))
        cp("$(py_dir)/custom_op.example", "custom_op.txt")
        @info "Edit custom_op.txt for custom operators"
        return
    else
        python = PyCall.python
        run(`$python $(py_dir)/customop.py custom_op.txt $py_dir false`)
    end
    nothing
end



function use_gpu(i::Union{Nothing,Int64}=nothing)
    if length(CUDA_INC)==0
        error("""ADCME is not built against GPU. Set ENV["GPU"]=1 and rebuild GPU.""")
    end
    dl = pyimport("tensorflow.python.client.device_lib")
    if !isnothing(i) && i>=1
        i = join(collect(0:i-1),',') 
        ENV["CUDA_VISIBLE_DEVICES"] = i 
    elseif !isnothing(i) && i==0
        ENV["CUDA_VISIBLE_DEVICES"] = ""
    end
    local_device_protos = dl.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]
end

function list_physical_devices(cpu_or_gpu::String = "all")
    dl = pyimport("tensorflow.python.client.device_lib")
    local_device_protos = dl.list_local_devices()
    CPU = [x.name for x in local_device_protos if x.device_type == "CPU"]
    GPU = [x.name for x in local_device_protos if x.device_type == "GPU"]
    if cpu_or_gpu == "all"
        return [CPU;GPU]
    elseif cpu_or_gpu == "GPU"
        return GPU 
    elseif cpu_or_gpu == "CPU"
        return CPU 
    else
        error(ArgumentError("$cpu_or_gpu is not a valid input. Expected: all, CPU, or GPU"))
    end
end




"""
    test_jacobian(f::Function, x0::Array{Float64}; scale::Float64 = 1.0)

Testing the gradients of a vector function `f`:
`y, J = f(x)` where `y` is a vector output and `J` is the Jacobian.
"""
function test_jacobian(f::Function, x0::Array{Float64}; scale::Float64 = 1.0)
    v0 = rand(Float64,size(x0))
    γs = scale ./10 .^(1:5)
    err2 = []
    err1 = []
    f0, J = f(x0)
    for i = 1:5
        f1, _ = f(x0+γs[i]*v0)
        push!(err1, norm(f1-f0))
        @show f1, f0, 2γs[i]*J*v0
        push!(err2, norm(f1-f0-γs[i]*J*v0))
        # push!(err2, norm((f1-f2)/(2γs[i])-J*v0))
        # #@show "test ", f1, f2, f1-f2
    end
    close("all")
    loglog(γs, err2, label="Automatic Differentiation")
    loglog(γs, err1, label="Finite Difference")
    loglog(γs, γs.^2 * 0.5*abs(err2[1])/γs[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
    loglog(γs, γs * 0.5*abs(err1[1])/γs[1], "--",label="\$\\mathcal{O}(\\gamma)\$")
    plt.gca().invert_xaxis()
    legend()
    println("Finite difference: $err1")
    println("Automatic differentiation: $err2")
    return err1, err2
end

"""
    install(s::String; force::Bool = false, islocal::Bool = false)

Install a custom operator from a URL, a directory (when `islocal` is true), or a string. In any of the three case, 
`install` copy the folder to $(abspath(joinpath(LIBADCME, "../../Plugin"))). 
When `s` is a string, `s` is converted to 

https://github.com/ADCMEMarket/<s>
"""
function install(s::String; force::Bool = false, islocal::Bool = false)
    if !islocal && !startswith(s, "http")
        s = "https://github.com/ADCMEMarket/"*s
    end 
    _, name = splitdir(s)
    if force 
        rm(joinpath(LIBPLUGIN, name), force=true, recursive=true)
    elseif isdir(joinpath(LIBPLUGIN, name)) && ("build" in readdir(joinpath(LIBPLUGIN, name)))
        return _plugin_lib(joinpath(LIBPLUGIN, name))
    end
    try
        LibGit2.clone(s, joinpath(LIBPLUGIN, name))
    catch
        LibGit2.clone("git://$(s[9:end]).git", joinpath(LIBPLUGIN, name))
    end
    PWD = pwd()
    cd(joinpath(LIBPLUGIN, name))
    if isfile(joinpath(LIBPLUGIN, name, "build.jl"))
        include(joinpath(LIBPLUGIN, name, "build.jl"))
    end
    cmakelists = String(read("CMakeLists.txt"))
    if !occursin("cmake_minimum_required", cmakelists)
        cmakelists = replace(String(read(joinpath(LIBPLUGIN, "CMakeLists.txt"))), "[INSTRUCTION]"=>cmakelists)
    end
    open("CMakeLists.txt", "w") do io 
        write(io, cmakelists)
    end
    mkdir("build")
    cd("build")
    ADCME.cmake()
    ADCME.make()
    cd(PWD)    
    return _plugin_lib(joinpath(LIBPLUGIN, name))
end

function _plugin_lib(D)
    files = readdir(joinpath(D, "build"))
    dylib = filter(x->endswith(x, ".$dlext"), files)
    if length(dylib)==0
        error(SystemError("No dynamic library found."))
    elseif length(dylib)>1
        error(SystemError("More then one dynamic libraries found."))
    else
        return joinpath(D, "build", dylib[1])
    end
end

function _make_blas(blas_binary)
    if Sys.iswindows()
        if isfile(joinpath(ADCME.LIBDIR, "openblas.lib"))
            return 
        end 
        if blas_binary
            @info "Downloading prebuilt blas from Github. If you encounter any problem with openblas when using adept, run `install_adept(blas_binary=false)` to compile from source"
            download("https://github.com/kailaix/tensorflow-1.15-include/releases/download/v0.1.0/openblas.lib", joinpath(ADCME.LIBDIR, "openblas.lib"))
            return 
        end
        @info "You are building openblas from source on Windows, and this process may take a long time.
Alternatively, you can place your precompiled binary to $(joinpath(ADCME.LIBDIR, "openblas.lib"))"
        PWD = pwd()
        download("https://github.com/xianyi/OpenBLAS/archive/v0.3.9.zip", joinpath(ADCME.LIBDIR, "OpenBlas.zip"))
        cd(ADCME.LIBDIR)
        run(`cmd /c unzip OpenBLAS.zip`)
        rm("OpenBlas", force=true, recursive=true)
        run(`cmd /c ren OpenBlas-0.3.9 OpenBlas`)
        rm("OpenBlas.zip")
        cd("OpenBlas")
        mkdir("build")
        cd("build")
        ADCME.cmake(CMAKE_ARGS="-DCMAKE_Fortran_COMPILER=flang -DBUILD_WITHOUT_LAPACK=no -DNOFORTRAN=0 -DDYNAMIC_ARCH=ON")
        ADCME.make()
        cd("../build/lib/Release")
        mv("openblas.lib", joinpath(ADCME.LIBDIR, "openblas.lib"))
        cd(PWD)
    else 
        required_file = Sys.isapple() ? ".dylib" : ".so"
        required_file = joinpath(ADCME.LIBDIR, "libopenblas")*required_file
        if !isfile(required_file)
            files = readdir(ADCME.LIBDIR)
            files = filter(x->!isnothing(x), match.(r"(libopenblas\S*.dylib)", files))[1]
            target = joinpath(ADCME.LIBDIR, files[1])
            symlink(target, required_file)
            @info "Symlink $(required_file) --> $(files[1])"
        end
    end 
end

"""
    install_adept(force::Bool=false)

Install adept-2 library: https://github.com/rjhogan/Adept-2
"""
function install_adept(force::Bool=false; blas_binary::Bool = true)
    PWD = pwd()
    cd(ADCME.LIBDIR)
    if force 
        @info "Removing Adept-2 by force..."
        rm("Adept-2", force=true, recursive=true)
    end
    if !isdir("Adept-2") 
        LibGit2.clone("https://github.com/ADCMEMarket/Adept-2", "Adept-2")
    end
    cd("Adept-2/adept")
    _make_blas(blas_binary)
    try
        if (!isfile("$(LIBDIR)/libadept.so") && !isfile("$(LIBDIR)/libadept.dylib") && !isfile("$(LIBDIR)/adept.lib")) || force
            @info """Copy "$(@__DIR__)/../deps/AdeptCMakeLists.txt" to "$(joinpath(pwd(), "CMakeLists.txt"))" ... """
            cp("$(@__DIR__)/../deps/AdeptCMakeLists.txt", "./CMakeLists.txt", force=true)
            @info """Remove $(joinpath(pwd(), "build")) ... """
            rm("build", force=true, recursive=true)
            @info "Make $(joinpath(pwd(), "build")) ... "
            mkdir("build")
            @info "Change directory into $(joinpath(pwd(), "build")) ... "
            cd("build")
            @info "Cmake ... "
            ADCME.cmake()
            @info "Make ... "
            ADCME.make()
        end
        printstyled("""
∘ Add the following lines to CMakeLists.txt 

include_directories(\${LIBDIR}/Adept-2/include)
find_library(ADEPT_LIB_FILE adept HINTS \${LIBDIR})
find_library(LIBOPENBLAS openblas HINTS \${LIBDIR})
message("ADEPT_LIB_FILE=\${ADEPT_LIB_FILE}")
message("LIBOPENBLAS=\${LIBOPENBLAS}")

∘ Add `\${ADEPT_LIB_FILE}` and `\${LIBOPENBLAS}` to `target_link_libraries`
""", color=:green)
    catch
        printstyled("Compliation failed\n", color=:red)
    finally
        cd(PWD)
    end
end

@doc raw"""
    register(forward::Function, backward::Function; multiple::Bool=false)

Register a function `forward` with back-propagated gradients rule `backward` to the backward. 
∘ `forward`: it takes $n$ inputs and outputs $m$ tensors. When $m>1$, the keyword `multiple` must be true. 
∘ `backward`: it takes $\tilde m$ top gradients from float/double output tensors of `forward`, $m$ outputs of the `forward`, 
   and $n$ inputs of the `forward`. `backward` outputs $n$ gradients for each input of `forward`. When input $i$ of
   `forward` is not float/double, `backward` should return `nothing` for the corresponding gradients. 
   
# Example 
```julia
forward = x->log(1+exp(x))
backward = (dy, y, x)->dy*(1-1/(1+y))
f = register(forward, backward)
```
"""
function register(forward::Function, backward::Function; multiple::Bool=false)
    fn_name = "customgrad_"*randstring(8)
    if !multiple
py"""
import tensorflow as tf
@tf.custom_gradient
def $$fn_name(*args):
    u = $forward(*args)
    def grad(dy):
        return $backward(dy, u, *args)
    return u, grad
"""
    else
py"""
import tensorflow as tf
@tf.custom_gradient
def $$fn_name(*args):
    u = forward_$$fn_name(*args)
    def grad(*dy):
        dy = [y for y in dy if y is not None and y.dtype in [tf.float64, tf.float32]] # only float64 and float32 can backpropagate gradients
        return backward_$$fn_name(*dy, *u, *args)
    return u, grad
"""
    end
    return py"$$fn_name"
end

"""
    debug(sess::PyObject, o::PyObject)

In the case a session run yields an error from the TensorFlow backend, this function can help print the exact error. 
For example, you might encounter  `InvalidArgumentError()` with no detailed error information, and this function can be useful for debugging.
"""
function debug(sess::PyObject, o::PyObject)
py"""
import tensorflow as tf
import traceback
try:
    $sess.run($o)
except Exception:
    print(traceback.format_exc())
"""
end

"""
    doctor()

Reports health of the current installed ADCME package. If some components are broken, possible fix is proposed.
"""
function doctor()
    function yes(name)
        printstyled("[✔️] $name\n", color=:green, bold=true)
    end
    function no(name, diagnose, instruction)
        printstyled("[✘] $name\n", color=:red, bold=true)
        printstyled("\n[Reason]\n", color=:magenta)
        printstyled("$diagnose\n\n", color=:blue)
        printstyled("\n[Instruction]\n", color=:magenta)
        printstyled("$instruction\n\n", color=:blue)
    end

    c = true 
    if VERSION>=v"1.4" && Sys.isapple() 
        c = false
    end

    if c 
        yes("Julia version")
    else
        no("Julia version", 
"""Your Julia version is $VERSION, and your system is MACOSX. This combination has a compatability issue.""",
"""Downgrade your Julia to ≦1.3""")
    end 

    c = (tf.__version__[1:6]=="1.15.0")
    if c 
        yes("TensorFlow version")
    else
        no("TensorFlow version", 
"""Your TensorFlow version is $(tf.__version__). ADCME is only tested against 1.15.0.""",
"""Set ENV["FORCE_REINSTALL_ADCME"] = 1 and rebuild ADCME
julia> ENV["FORCE_REINSTALL_ADCME"] = 1
julia> ]
pkg> build ADCME""")
    end 

    c = (tfp.__version__[1:5]=="0.8.0")
    if c 
        yes("TensorFlow-Probability version")
    else
        no("TensorFlow-Probability version", 
"""Your TensorFlow-Probability version is $(tfp.__version__). ADCME is only tested against 0.8.0.""",
"""Set ENV["FORCE_REINSTALL_ADCME"] = 1 and rebuild ADCME
julia> ENV["FORCE_REINSTALL_ADCME"] = 1
julia> ]
pkg> build ADCME""")
    end 



    c = (PyCall.python==ADCME.PYTHON)
    if c 
        yes("Python executable file")
    else
        no("Python executable file", 
"""PyCall Python path $(PyCall.python) does not match the ADCME-compatible Python $(ADCME.PYTHON)""",
"""Rebuild PyCall with a compatible Python version:

using Pkg
ENV["PYTHON"] = "$(ADCME.PYTHON)"
Pkg.build("PyCall")
""")
    end 

    c = true 
    try 
        if Sys.iswindows()
            run(`cmd /c where julia`)
        else 
            run(`which julia`)
        end
    catch
        c = false
    end

    if c 
        yes("Julia path")
    else
        no("Julia path (Optional)", 
"""`julia` outputs nothing. This will break custom operator compilation.""",
"""Add your julia binary path to your environment path, e.g. (Unix systems) 

export PATH=$(Sys.BINDIR):\$PATH

For convenience, you can add the above line to your `~/.bashrc` (Linux) or `~/.bash_profile` (Apple).
For Windows, you need to add it to system environment.""")
    end

    c = Sys.iswindows() ?
        haskey(ENV, "PATH") && occursin(ADCME.LIBDIR, ENV["PATH"]) :
        haskey(ENV, "LD_LIBRARY_PATH") && occursin(ADCME.LIBDIR, ENV["LD_LIBRARY_PATH"])
    if c 
        yes("Dynamic library path")
    else
        no("Dynamic library path (Optional)", 
"""$(ADCME.LIBDIR) is not in LD_LIBRARY_PATH. This MAY break custom operator compilation. However, in most cases, ADCME automatic fixes this problem for you.""",
"""Add your dynamic library path path to your environment path, e.g. (Unix systems) 

export LD_LIBRARY_PATH=$(ADCME.LIBDIR):\$LD_LIBRARY_PATH

For convenience, you can add the above line to your `~/.bashrc` (Linux or Apple).
For Windows, you need to add it to PATH instead of LD_LIBRARY_PATH.""")
    end

    c  = Sys.WORD_SIZE==64
    if c 
        yes("Memory Address Length =  64")
    else 
        no("Memory Address Length",
"""Your memory address length is $(Sys.WORD_SIZE). ADCME is only tested against 64-bit machine.""",
"""If you do not need custom operators, then it's fine. Otherwise you need to switch to a 64-bit machine""")
    end

    if Sys.iswindows()
        c = isfile(ADCME.MAKE*".exe") && occursin("15", (ADCME.MAKE)) && occursin("2017", ADCME.MAKE)
        if c 
            yes("C Compiler")
        else
            no("C Compiler", 
"""You specified that the C compiler for custom operators is 
$(ADCME.MAKE)
However, one of the following requirements is not met: 
1*. The file you specified $(ADCME.MAKE*".exe") does not exist.
2**. (Optional) For compatibility, we suggest you use Microsoft Visual Studio 2017 (Version number: 15).

* The path is actually not needed in compilation, but we raise such an issue here in case you obtain some compilation errors in the future.

* We check the version by looking for "15" and "2017" in the path specification. If you are sure your compiler is correct, you can ignore this message. """,
"""Manually edit $(abspath(joinpath(splitdir(pathof(ADCME))[1], "../deps/deps.jl"))) and modify `MAKE` to be the correct compiler.""")
        end 

    end
    
    

    c = haskey(ENV, "PATH") && occursin(ADCME.BINDIR, ENV["PATH"])
    if c 
        yes("Binaries path")
    else
        no("Binaries path", 
"""$(ADCME.BINDIR) is not in PATH. This path contains compatible tools such as a GCC compiler, `cmake`, `make`, or any other tools you want to use directly from terminal.
However, setting the path is NOT a requirement, and ADCME works totally fine without any action.""",
"""(Optional) Add your binary path to your environment path, e.g. (Unix systems) 

export PATH=$(ADCME.BINDIR):\$PATH

For convenience, you can add the above line to your `~/.bashrc` (Linux) or `~/.bash_profile` (Apple).
For Windows, you need to add it to system environment.""")
    end

    if length(ADCME.CUDA_INC)>0
        c = Sys.iswindows() ?
                haskey(ENV, "PATH") && occursin(ADCME.LIBCUDA, ENV["PATH"]) :
                haskey(ENV, "LD_LIBRARY_PATH") && occursin(ADCME.LIBCUDA, ENV["LD_LIBRARY_PATH"])
        if c 
            yes("CUDA LD_LIBRARY_PATH")
        else
            no("CUDA LD_LIBRARY_PATH", 
    """$(ADCME.LIBCUDA) is not in LD_LIBRARY_PATH. This path contains compatible tools such as a GCC compiler, `cmake`, `make`, etc.""",
    """The fix is OPTIONAL.
    Add your binary path to your environment path, e.g. (Unix systems) 
    
    export LD_LIBRARY_PATH=$(ADCME.LIBCUDA):\$LD_LIBRARY_PATH
    
    For convenience, you can add the above line to your `~/.bashrc` (Linux or Apple).
    For Windows, you need to add it to PATH instead of LD_LIBRARY_PATH.""")
        end

        try 
            if !Sys.iswindows()
                Libdl.dlpath("libcuda")
                Libdl.dlpath("libcudnn")
                Libdl.dlpath("libcublas")
            else
                Libdl.dlpath("cudart64_100")
                Libdl.dlpath("cudnn64_7")
                Libdl.dlpath("cublas64_100")
            end
            yes("CUDA Shared Library")
        catch
            no("CUDA Shared Library", 
    """libcuda, libcudnn, and (or) libcublas can not be loaded.""",
    """If you intend to use GPU, this fix is mandatory. Make sure cudatoolkit and cudnn libraries can be found in
$(ADCME.LIBCUDA)
and `nvcc` is in your path.""")
        end

        c = isdir(ADCME.CUDA_INC) && "cuda.h" in readdir(ADCME.CUDA_INC)
        if c 
            yes("CUDA Header Files")

            if !isfile(joinpath(ADCME.TF_INC, "third_party/gpus/cuda/include/cuda_fp16.h"))
                println("Fixing third_party/gpus/cuda/include...")
                if !ispath(joinpath(ADCME.TF_INC, "third_party/gpus/cuda/"))
                    mkpath(joinpath(ADCME.TF_INC, "third_party/gpus/cuda/"))
                end
                rm(joinpath(ADCME.TF_INC, "third_party/gpus/cuda/include/"), force=true, recursive=true)
                symlink(ADCME.CUDA_INC, joinpath(ADCME.TF_INC, "third_party/gpus/cuda/include"))
            end

        else 
            no("CUDA Header Files",
            """Cuda include library does not exist or `cuda.h` is missing.""",
        """It might be possible that your cuda include library is located somewhere else other than $(ADCME.CUDA_INC). Fix the dependency file.""")
        end
    else
        no("GPU Support (Optional)", 
    """ADCME is not compiled against GPU.""",
    """If you intend to use GPU, set ENV["GPU"] = 1 and then rebuild ADCME.""")
    end

    

    println("Dependency file is located at: $(joinpath(@__DIR__, "../deps/deps.jl"))")
    
end


"""
    test_gpu()

Tests the GPU ultilities
"""
function test_gpu()
    PWD = pwd()
    run(`which nvcc`)
    cd("$(@__DIR__)/../examples/gpu_custom_op")
    mkdir("build")
    cd("build")
    ADCME.cmake()
    ADCME.make()
    cd("..")
    include("gputest.jl")
    cd(PWD)
end


@doc raw"""
    nnuq(H::Array{Float64,2}, invR::Union{Float64, Array{Float64,2}}, invQ::Union{Float64, Array{Float64,2}})

Returns the variance matrix for the Baysian inversion. 

The negative log likelihood function is

$$l(s) =\frac{1}{2} (y-h(s))^T R^{-1} (y-h(s)) + \frac{1}{2} s^T Q^{-1} s$$

The covariance matrix is computed by first linearizing $h(s)$

$$h(s)\approx h(s_0) + \nabla h(s_0) (s-s_0)$$

and then computing the second order derivative

$$V = \left(\frac{\partial^2 l}{\partial s^T\partial s}\right)^{-1} = (H^T R^{-1} H + Q^{-1})^{-1}$$

Note the result is independent of $s_0$, $y_0$, and only depends on $\nabla h(s_0)$
"""
function nnuq(H::Array{Float64,2}, invR::Union{Float64, Array{Float64,2}}, invQ::Union{Float64, Array{Float64,2}})
    if isa(invQ, Float64)
        invQ = invQ * I 
    end
    Σ = inv(H' * invR * H + invQ)
    (Σ + Σ')/2
end



mutable struct MCMCSimple
    logf::Function
    proposal::Function
    θ0::Array{Float64, 1}
    ub::Float64
    lb::Float64 
    L::Array{Float64, 1}
    AC::Array{Float64, 1}
end

@doc raw"""
    MCMCSimple(obs::Array{Float64, 1}, h::Function, 
    σ::Float64, θ0::Array{Float64,1}, lb::Float64, ub::Float64)

A very simple yet useful interface for MCMC simulation in many scientific computing problems. 

- `obs`: Observations
- `h`: Forward computation function
- `σ`: Noise standard deviation for the observed data 
- `ub`, `lb`: upper and lower bound
- `θ0`: Initial guess 

The mathematical model is 

```math
y_{obs} = h(\theta)
```

and we have a hard constraint `lb\leq \theta \leq ub`. 
"""
function MCMCSimple(obs::Array{Float64, 1}, h::Function, 
        σ::Float64, θ0::Array{Float64,1}, lb::Float64, ub::Float64,
        δ::Union{Missing, Float64}=missing)
    τ = (ub-lb)/6
    δ = coalesce(δ, (ub-lb)/100)
    function logf(x)
        -sum((obs - h(x)).^2/2σ^2) - sum((x-θ0).^2)/2τ^2
    end
    function proposal(x)
        x + (rand(length(x)) .- 0.5)*2 * δ
    end
    MCMCSimple(logf, proposal, θ0, ub, lb, zeros(0), zeros(0))
end

function simulate(ms::MCMCSimple, N::Int64, burnin::Union{Int64, Missing} = missing)
    burnin = coalesce(burnin, Int64(round(N*0.2)))
    sim = zeros(N, length(ms.θ0))
    sim[1,:] = ms.θ0
    L = zeros(N)
    AC = ones(N)
    AC[1] = NaN
    L[1] = ms.logf(ms.θ0)
    k = 1
    for i = 2:N 
        sim[i,:], L[i], k_ = _MCMCSimple_simulate(ms, sim[i-1,:])
        k += k_ 
        AC[i] = k/i
    end
    ms.AC = AC
    ms.L = L 
    return sim
end

function diagnose(ms::MCMCSimple)
    if !isdefined(Main, :PyPlot)
        error("Package PyPlot.jl must be imported in the main module using `import PyPlot` or `using PyPlot`")
    end
    Main.PyPlot.figure(figsize = (10, 4))
    Main.PyPlot.subplot(121)
    Main.PyPlot.title("Acceptance Rate")
    Main.PyPlot.plot(ms.AC)
    Main.PyPlot.ylim(0,1.05)
    Main.PyPlot.subplot(122)
    Main.PyPlot.title("Log Likelihood")
    Main.PyPlot.plot(ms.L)
end

function _MCMCSimple_simulate(ms::MCMCSimple, x::Array{Float64})
    local x_star
    while true
        x_star = ms.proposal(x)
        if all(x_star.<=ms.ub) && all(x_star.>=ms.lb)
            break
        end
    end
    Δ =  ms.logf(x_star) - ms.logf(x)
    if log(rand())<Δ
        return x_star, ms.logf(x_star), 1
    else 
        return x, ms.logf(x), 0
    end
end