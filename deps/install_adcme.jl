# check if ADCME is installed
CONDA = ""
if Sys.iswindows()
    CONDA = "$(homedir())/.julia/conda/3/Scripts/conda.exe"
else 
    CONDA = "$(homedir())/.julia/conda/3/bin/conda"
end

if (!FORCE_REINSTALL_ADCME) && isfile(CONDA) && occursin("4.8.4", read(`$CONDA --version`, String)) && occursin("tensorflow", read(`$CONDA list`, String))
    @info "ADCME dependencies have already been installed"
else
    installer = ""
    if Sys.islinux() 
        installer = "Miniconda3-py37_4.8.3-Linux-x86_64.sh"
    elseif Sys.isapple()
        installer = "Miniconda3-py37_4.8.3-MacOSX-x86_64.sh"
    else 
        installer = "Miniconda3-py37_4.8.3-Windows-x86_64.exe"
    end

    if !isdir("$(homedir())/.julia/conda/")
        mkpath("$(homedir())/.julia/conda/")
    end 

    PWD = pwd()
    cd("$(homedir())/.julia/conda/")
    if !(installer in readdir("."))
        @info "Downloading miniconda installer..."
        download("https://repo.anaconda.com/miniconda/"*installer, installer)
    end
    if isdir("3")
        mv("3", "trash", force=true)
    end
    @info "Installing miniconda..."
    if Sys.iswindows()
        run(`cmd /c start /wait "" $installer /InstallationType=JustMe /RegisterPython=0 /S /D=$(homedir())\\.julia\\conda\\3`)
    else
        run(`bash $installer -b -p 3`)
    end
    cd(PWD)

    if Sys.iswindows()
        run(`$CONDA config --add channels conda-forge`)
        run(`$CONDA install -y matplotlib=3.3.0 pandas=1.1.0
            numpy=1.18.5 ninja=1.10  tensorflow=1.15 tensorflow-probability=0.8 
            blas=1.0`)
    elseif Sys.islinux()
        run(`$CONDA config --add channels conda-forge`)
        run(`$CONDA install -y matplotlib=3.3.0 pandas=1.1.0 bazel=0.26.1 
            numpy=1.18.5 ninja=1.10 make=4.3 lapack=3.6.1 unzip=6.0 tensorflow=1.15 tensorflow-probability=0.8 openblas=0.3.10 
            gcc_linux-64=5.4.0 gxx_linux-64=5.4.0`)
    else 
        run(`$CONDA config --add channels conda-forge`)
        run(`$CONDA install -y matplotlib=3.3.0 pandas=1.1.0 bazel=0.26.1 
            numpy=1.18.5 ninja=1.10 make=4.3 lapack=3.6.1 unzip=6.0 tensorflow=1.15 tensorflow-probability=0.8 openblas=0.3.10 
            clang=4.0.1 clangxx=4.0.1`)
    end
end
