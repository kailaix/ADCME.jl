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
uqlin, 
uqnlin,
clean

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
    if haskey(ENV_, "LD_LIBRARY_PATH")
        ENV_["LD_LIBRARY_PATH"] = ENV["LD_LIBRARY_PATH"]*":$LIBDIR"
    else
        ENV_["LD_LIBRARY_PATH"] = LIBDIR
    end
    if Sys.iswindows()
        run(setenv(`$CMAKE -G"Visual Studio 15" -DJULIA="$(joinpath(Sys.BINDIR, "julia"))" -A x64 $CMAKE_ARGS $DIR`, ENV_)) # very important, x64
    else
        run(setenv(`$CMAKE -DJULIA="$(joinpath(Sys.BINDIR, "julia"))" -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX $CMAKE_ARGS $DIR`, ENV_))
    end
end

function make()
    ENV_ = copy(ENV)
    if haskey(ENV_, "LD_LIBRARY_PATH")
        ENV_["LD_LIBRARY_PATH"] = ENV["LD_LIBRARY_PATH"]*":$LIBDIR"
    else
        ENV_["LD_LIBRARY_PATH"] = LIBDIR
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
        run(`cmd /c $CMAKE --build . --target ALL_BUILD --config Release`)
    else
        run(setenv(`$MAKE -j`, ENV_))
    end
end

load_op_dict = Dict{Tuple{String, String}, PyObject}()
load_op_grad_dict = Dict{Tuple{String, String}, PyObject}()


@doc """
    load_op(oplibpath::String, opname::String)

Loads the operator `opname` from library `oplibpath`.
"""
function load_op(oplibpath::String, opname::String)
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
        s = py"$$fn_name"
        load_op_grad_dict[(oplibpath,opname)] = s
        printstyled("Load library operator (with gradient, multiple outputs = $multiple): $oplibpath ==> $opname\n", color=:green)
        return s
end

"""
    clean()

Clean up CustomOps directory. 
"""
function clean()
    colibs = include("$(@__DIR__)/../deps/CustomOps/default_formulas.jl")
    CO = String[]
    for c in colibs
        push!(CO, c.second[1])
    end
    for c in CO 
        rm(joinpath("$(@__DIR__)/../deps/CustomOps/"*c, "build"), force=true, recursive=true)
    end
    for file in readdir("$(@__DIR__)/../deps/CustomOps/")
        if isdir(file) && !(file in CO)
            rm(file, recursive=true, force=true)
        end
    end
    rm("$(@__DIR__)/../deps/CustomOps/formulas.txt", force=true)
    rm("$(@__DIR__)/../deps/CustomOps/CMakeLists.txt", force=true)
    rm("$(@__DIR__)/../deps/CustomOps/build", recursive=true, force=true)
end

function refresh_cmake()
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
    
    NEWLIB = String[]
    for c in COLIB
        push!(NEWLIB, c.second[1])
    end
    NEWLIB = unique(NEWLIB)
    cmakecnt = read(joinpath(@__DIR__, "../deps/CustomOps/CMakeListsTemplate.txt"),String)
    cmakecnt = replace(cmakecnt, "set(LIBDIR_NAME"=>"set(LIBDIR_NAME "*join(NEWLIB, ' '))
    open(joinpath(@__DIR__, "../deps/CustomOps/CMakeLists.txt"), "w") do io 
        write(io, cmakecnt)
    end
    NEWLIB
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
function load_system_op(s::String, oplib::String, opname::String, grad::Bool=true; 
    return_str::Bool=false, multiple::Bool=false)
    dir = joinpath(joinpath("$(@__DIR__)", "../deps/CustomOps"), s)
    if !isdir(dir)
        error("Folder for the operator $s does not exist: $dir")
    end
    if Sys.iswindows()
        if length(oplib)>=3 && oplib[1:3]=="lib"
            oplib = oplib[4:end]
        end
    end
    oplibpath = joinpath(joinpath(dir, "build"), oplib)
    # check if the library exists 
    libfile = abspath(oplibpath * (Sys.islinux() ? 
                        ".so" : Sys.isapple() ? ".dylib" : ".dll"))
    @show libfile
    # error()
    if !isfile(libfile)
        @info "Lib $s exists in registery but was not initialized. Compiling..."
        ADCME.precompile()
    end
    if return_str
        return oplibpath
    end
    if grad
        load_op_and_grad(oplibpath, opname; multiple=multiple)
    else
        load_op(oplibpath, opname)
    end
end

load_system_op(s::String; kwargs...) = load_system_op(COLIB[s]...; kwargs...)

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
    if (!force) && isfile("$(@__DIR__)/../deps/CustomOps/CMakeLists.txt") && 
            isdir("$(@__DIR__)/../deps/CustomOps/build")
        @info "Reuse existing cmake files"
        PWD = pwd()
        cd("$(@__DIR__)/../deps/CustomOps/build")
        ADCME.cmake()
        ADCME.make()
        cd(PWD)
        return 
    end
    refresh_cmake()
    cd("$(@__DIR__)/../deps/CustomOps")
    rm("build", force=true, recursive=true)
    mkdir("build")
    cd("build")
    ADCME.cmake()
    ADCME.make()
    cd(PWD)
end


"""
    customop(simple::Bool=false)

Create a new custom operator. If `simple=true`, the custom operator only supports CPU and does not have gradients. 

# Example

```julia-repl
julia> customop() # create an editable `customop.txt` file
[ Info: Edit custom_op.txt for custom operators
julia> customop() # after editing `customop.txt`, call it again to generate interface files.
```
"""
function customop(simple::Bool=false)
    # install_custom_op_dependency()
    py_dir = "$(@__DIR__)/../examples/custom_op/template"
    if !("custom_op.txt" in readdir("."))
        cp("$(py_dir)/custom_op.example", "custom_op.txt")
        @info "Edit custom_op.txt for custom operators"
        return
    else
        python = PyCall.python
        run(`$python $(py_dir)/customop.py custom_op.txt $py_dir $simple`)
    end
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
    install(s::String; force::Bool = false)

Install a custom operator via URL. `s` can be
- A URL. ADCME will download the directory through `git`
- A string. ADCME will search for the associated package on https://github.com/ADCMEMarket
"""
function install(s::String; force::Bool = false)
    global COLIB
    codir = "$(@__DIR__)/../deps/CustomOps"
    if !startswith(s, "https://github.com")
        s = "https://github.com/ADCMEMarket/"*s
    end
    _, name = splitdir(s)
    if name in readdir(codir)
        if force
            rm(joinpath(codir, name), recursive=true, force=true)
        else
            error("$name already in $codir, fix it with\n\n\tinstall(\"$s\", force=true)\n")
        end
    end
    try
        LibGit2.clone(s, joinpath(codir, name))
    catch
        LibGit2.clone("git://$(s[9:end]).git", joinpath(codir, name))
    end

    # If there is a build script `build.sh`, run the build script
    if isfile(joinpath(joinpath(codir, name), "build.sh"))
        PWD = pwd()
        cd(joinpath(codir, name))
        try
            run(`bash build.sh`)
        catch 
        end
        cd(PWD)
    end

    formula = eval(Meta.parse(read(joinpath(joinpath(codir, name),"formula.txt"), String)))
    if isnothing(formula)
        error("Broken package: $s does not have formula.txt.")
    else
        @info "Add formula $formula"
        push!(COLIB, formula)
    end

    rm("$(@__DIR__)/../deps/CustomOps/formulas.jl", force=true)
    open("$(@__DIR__)/../deps/CustomOps/formulas.jl", "a") do io 
        n = length(COLIB)
        for (k, c) in enumerate(COLIB)
            if k == n 
                write(io, string(c)*"\n")
            else 
                write(io, string(c)*",\n")
            end
        end
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
message("ADEPT_LIB_FILE=\${ADEPT_LIB_FILE}")

∘ Add `\${ADEPT_LIB_FILE}` to `target_link_libraries`
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

In the case a session run yields `InvalidArgumentError()`, this function can help print the exact error. 
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
        run(`which julia`)
    catch
        c = false
    end

    if c 
        yes("Julia path")
    else
        no("Julia path", 
"""`which julia` outputs nothing. This will break custom operator compilation.""",
"""Add your julia binary path to your environment path, e.g. (Unix systems) 

export PATH=$(Sys.BINDIR):\$PATH

For convenience, you can add the above line to your `~/.bashrc` (Linux) or `~/.bash_profile` (Apple).
For Windows, you need to add it to system environment.""")
    end

    c = haskey(ENV, "LD_LIBRARY_PATH") && occursin(ADCME.LIBDIR, ENV["LD_LIBRARY_PATH"])
    if c 
        yes("Dynamic library path")
    else
        no("Dynamic library path", 
"""$(ADCME.LIBDIR) is not in LD_LIBRARY_PATH. This MAY break custom operator compilation. However, in most cases, ADCME automatic fixes this problem for you.""",
"""(Optional) Add your dynamic library path path to your environment path, e.g. (Unix systems) 

export LD_LIBRARY_PATH=$(ADCME.LIBDIR):\$LD_LIBRARY_PATH

For convenience, you can add the above line to your `~/.bashrc` (Linux) or `~/.bash_profile` (Apple).
For Windows, you need to add it to system environment.""")
    end

    c  = Sys.WORD_SIZE==64
    if c 
        yes("Memory Address Length")
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
        c = haskey(ENV, "LD_LIBRARY_PATH") && occursin(ADCME.LIBCUDA, ENV["LD_LIBRARY_PATH"])
        if c 
            yes("CUDA LD_LIBRARY_PATH")
        else
            no("CUDA LD_LIBRARY_PATH", 
    """$(ADCME.LIBCUDA) is not in LD_LIBRARY_PATH. This path contains compatible tools such as a GCC compiler, `cmake`, `make`, etc.""",
    """The fix is OPTIONAL.
    Add your binary path to your environment path, e.g. (Unix systems) 
    
    export LD_LIBRARY_PATH=$(ADCME.LIBCUDA):\$LD_LIBRARY_PATH
    
    For convenience, you can add the above line to your `~/.bashrc` (Linux) or `~/.bash_profile` (Apple).
    For Windows, you need to add it to system environment.""")
        end

        try 
            Libdl.dlpath("libcuda")
            Libdl.dlpath("libcudnn")
            Libdl.dlpath("libcublas")
            yes("CUDA Library")
        catch
            no("CUDA Library", 
    """libcuda, libcudnn, and (or) libcublas can not be loaded.""",
    """If you intend to use GPU, this fix is mandatory. Make sure cudatoolkit and cudnn libraries can be found in
$(ADCME.LIBCUDA)
and `nvcc` is in your path.""")
        end

        c = isdir(ADCME.CUDA_INC) && "cuda.h" in readdir(ADCME.CUDA_INC)
        if c 
            yes("CUDA Include Library")

            if !isfile(joinpath(ADCME.TF_INC, "third_party/gpus/cuda/include/cuda_fp16.h"))
                println("Fixing third_party/gpus/cuda/include...")
                if !ispath(joinpath(ADCME.TF_INC, "third_party/gpus/cuda/"))
                    mkpath(joinpath(ADCME.TF_INC, "third_party/gpus/cuda/"))
                end
                rm(joinpath(ADCME.TF_INC, "third_party/gpus/cuda/include/"), force=true, recursive=true)
                symlink(ADCME.CUDA_INC, joinpath(ADCME.TF_INC, "third_party/gpus/cuda/include"))
            end

        else 
            no("CUDA Include Library",
            """Cuda include library does not exist or `cuda.h` is missing.""",
        """It might be possible that your cuda include library is located somewhere else other than $(ADCME.CUDA_INC). Fix the dependency file.""")
        end
    else
        no("GPU Support", 
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
    uqlin(y::Array{Float64,1}, 
    H::Array{Float64,2},
    R::Array{Float64,2},
    X::Array{Float64},
    Q::Array{Float64,2})

Geeneral linear estimation. The model is 

$$\begin{aligned}
y &= H s + \delta\\
s &= X\beta + \epsilon
\end{aligned}$$

where 

$$\delta \sim \mathcal{N}(0, R)\quad \epsilon \sim \mathcal{N}(0, Q)$$

Inputs:

- `y`: length $n$ measurement vector 
- `H`: $n\times m$ observation matrix 
- `R`: $n\times n$ measurement covariance 
- `X`: $m\times p$ drift matrix for `s`
- `Q`: $m\times m$ hidden parameter covariance 

Outputs:

- `s`: length $m$ posterial mean 
- `Σ`: $m\times m$ posterial covariance 

!!! info 
    
"""
function uqlin(y::Array{Float64,1}, 
    H::Array{Float64,2},
    R::Array{Float64,2},
    X::Array{Float64},
    Q::Array{Float64,2})
    n, m = size(H)
    if length(size(X))==1
        X = reshape(X, :, 1)
    end
    if length(y)!=n || 
        size(R)!=(n,n) || 
        size(Q)!=(m,m) ||
        size(X,1)!=m 
        error("Check dimension")
    end
    p = size(X,2)

    PHI = H*X; QHT = Q*H'; HQHT = H*QHT;
    PSI = HQHT+R;

    AA = [PSI PHI
         PHI' zeros(p,p)]
    AA=(AA+AA')/2;
    bb = [QHT';X'];

    SOL = AA\bb; 
    LAMBDA = (SOL[1:n,:])'; 
    MU = SOL[n+1:n+p,:];

    s = LAMBDA*y; 
    V = -X*MU+Q-QHT*LAMBDA';
    V = (V+V')/2
    return s, V
end

@doc raw"""
    uqnlin(
        y::Array{Float64,1}, 
        hs::Array{Float64},
        H::Array{Float64,2},
        R::Array{Float64,2},
        s::Array{Float64},
        Q::Array{Float64,2}
    )

Uncertainty quantification for linearized Gaussian model. 
Similar to [`uqlin`](@ref) except that 
we solve a nonlinear system 
```math
$$\begin{aligned}
y &= h(s) + \delta \\ 
s &= g(z) + \epsilon
\end{aligned}\tag{1}$$
```
"""
function uqnlin(
    y::Array{Float64,1}, 
    hs::Array{Float64},
    H::Array{Float64,2},
    R::Array{Float64,2},
    s::Array{Float64},
    Q::Array{Float64,2}
)
    y = y - (hs-H*s)
    X = s
    μ, Σ = uqlin(y, H, R, X, Q)
end
