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
export http_file, http_archive, git_repository, require_file, 
    link_file, make_directory, change_directory, require_library, get_library

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

function http_archive(url::AbstractString, file::AbstractString)
    file = abspath(file)
    d, _ = splitdir(file)
    _, e = splitext(url)
    if !(e in [".zip", ".tar", ".tar.gz", "tar.bz2"])
        error("The extension of the archive $e not understood")
    end
    zipname = splitdir(url)[2]
    zipname = joinpath(d, zipname)
    require_file(file) do 
        require_file(zipname) do 
            download(url, zipname)
        end
        if e==".zip"
            run(`unzip `)
        elseif e==".tar"
        elseif e==".tar.gz"
        elseif e=="tar.bz2"
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