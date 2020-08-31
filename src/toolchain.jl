#=
toolchain.jl implements a build system 

Compilers
=========
CC 
CXX
GFORTRAN 

Tools 
=====
CMAKE
MAKE
NINJA

Directories
===========
BINDIR
LIBDIR
PREFIXDIR
=#
export http_file, uncompress, git_repository, require_file, 
    link_file, make_directory, change_directory, require_library, get_library,
    run_with_env

GFORTRAN = nothing

"""
    http_file(url::AbstractString, file::AbstractString)

Download a file from `url` and rename it to `file`.
"""
function http_file(url::AbstractString, file::AbstractString)
    require_file(file) do 
        @info "Downloading $url -> $file"
        download(url, file)
        @info "Downloaded $file from $url."
    end
end

"""
    uncompress(zipfile::AbstractString, file::AbstractString)

Uncompress a zip file `zipfile` to `file` (a directory). Note this function does not check that the 
uncompressed content has the name `file`. Users may use `mv uncompress_file file` to enforce the consistency.
"""
function uncompress(zipfile::AbstractString, file::Union{Missing, AbstractString}=missing)
    zipfile = abspath(zipfile)
    if ismissing(file)
        d = "."
    else
        file = abspath(file)
        d = splitdir(file)
    end
    uncompress_ = ()->begin
        if length(zipfile)>4 && zipfile[end-3:end]==".zip"
            run(`unzip $zipfile -d $d`)
        elseif length(zipfile)>4 && zipfile[end-3:end]==".tar"
            run(`tar -xvf $zipfile -C $d`)
        elseif length(zipfile)>7 && zipfile[end-6:end]==".tar.gz"
            run(`tar -xvzf $zipfile -C $d`)
        else 
            error("ADCME doesn't know how to uncompress $zipfile")
        end
    end
    if ismissing(file)
        uncompress_()
    else 
        require_file(file) do 
            uncompress_()
        end
    end
end

"""
    git_repository(url::AbstractString, file::AbstractString)

Clone a repository `url` and rename it to `file`.
"""
function git_repository(url::AbstractString, file::AbstractString)
    @info "Cloning from $url to $file..."
    require_file(file) do 
        LibGit2.clone(url, file)
        @info "Cloned $url to $file"
    end
end

"""
    require_file(f::Function, file::Union{String, Array{String}})

If any of the files/links/directories in `file` does not exist, execute `f`.
"""
function require_file(f::Function, file::Union{String, Array{String}})
    if isa(file, String)
        file = [file]
    end
    if !all([isfile(x)||islink(x)||isdir(x) for x in file])
        f()
    else
        if length(file)==1
            @info "File $(file[1]) exists"
        else
            @info "Files exist: $(file)"
        end
    end
end

"""
    require_gfortran()

Install a gfortran compiler if it does not exist.
"""
function require_gfortran()
    global GFORTRAN
    try 
        GFORTRAN = split(String(read(`which gfortran`)))[1]
    catch
        error("gfortran is not in the path.")
    end
end

"""
    link_file(target::AbstractString, link::AbstractString)

Make a symbolic link `link` -> `target`
"""
function link_file(target::AbstractString, link::AbstractString)
    if isfile(link) || islink(link)
        return 
    else
        symlink(target, link)
        @info "Made symbolic link $link -> $target"
    end
end

"""
    make_directory(directory::AbstractString)

Make a directory if it does not exist. 
"""
function make_directory(directory::AbstractString)
    require_file(directory) do 
        mkdir(directory)
        @info "Made directory directory"
    end
end

"""
    change_directory(directory::AbstractString)

Change the current working directory to `directory`. If `directory` does not exist, it is made. 
"""
function change_directory(directory::AbstractString)
    if !isdir(directory)
        make_directory(directory)
    end
    cd(directory)
    @info "Changed to directory $directory"
end

"""
    get_library(filename::AbstractString)

Returns a valid library file. For example, for `filename = adcme`, we have 

- On MacOS, the function returns `libadcme.dylib`
- On Linux, the function returns `libadcme.so`
- On Windows, the function returns `adcme.dll`
"""
function get_library(filename::AbstractString)
    filename = abspath(filename)
    dir, file = splitdir(filename)
    if Sys.islinux() || Sys.isapple()
        if length(file)<3 || file[1:3]!="lib"
            file = "lib" * file 
        end
        f, _ = splitext(file)
        ext = Sys.islinux() ? "so" : "dylib"
        file = f * "." * ext 
    else
        f, _ = splitext(file)
        file = f * ".dll"
    end
    filename = joinpath(dir, file)
end

"""
    require_library(func::Function, filename::AbstractString)

If the library file `filename` does not exist, `func` is executed.
"""
function require_library(func::Function, filename::AbstractString)
    filenmae = get_library(filename)
    if !(isfile(filename) && islink(filename))
        func()
    end
end

"""
    run_with_env(cmd::Cmd, env::Union{Missing, Dict} = missing)

Running the command with the default environment and an extra environment variables `env`
"""
function run_with_env(cmd::Cmd, env::Union{Missing, Dict} = missing)
    ENV_ = copy(ENV)
    LD_PATH = Sys.iswindows() ? "PATH" : "LD_LIBRARY_PATH"
    if haskey(ENV_, LD_PATH)
        ENV_[LD_PATH] = ENV[LD_PATH]*":$LIBDIR"
    else
        ENV_[LD_PATH] = LIBDIR
    end
    if !ismissing(env)
        ENV_ = merge(env, ENV_)
    end
    run(setenv(cmd, ENV_))
end