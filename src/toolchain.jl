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

GFORTRAN = nothing


function http_file(url::AbstractString, file::AbstractString)
    require_file(file) do 
        download(url, file)
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

function git_repository(url::AbstractString, file::AbstractString)
    require_file(file) do 
        LibGit2.clone(url, file)
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
    end
end

"""
    require_gfortran()

Install a gfortran compiler if it does not exist.
"""
function require_gfortran()
    global GFORTRAN
    if isnothing(GFORTRAN)
        CONDA = joinpath(BINDIR, "conda")
        require_file(joinpath(BINDIR, "gfortran")) do 
            if Sys.islinux()
                run(`$CONDA install --no-update-dependencies -y -c anaconda gfortran_linux-64`)
            elseif Sys.isapple()
                run(`$CONDA install --no-update-dependencies -y -c anaconda gfortran_osx-64`)
            end
        end
        GFORTRAN = joinpath(BINDIR, "gfortran")
    end
end

function link_file(target::AbstractString, link::AbstractString)
    if isfile(link) || islink(link)
        return 
    else
        symlink(target, link)
    end
end