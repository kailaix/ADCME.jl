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
        try
            mv("3", "trash", force=true)
        catch 
        end
    end
    @info "Installing miniconda..."
    if Sys.iswindows()
        run(`cmd /c start /wait "" $installer /InstallationType=JustMe /RegisterPython=0 /S /D=$(homedir())\\.julia\\conda\\3`)
    else
        run(`bash $installer -b -p 3`)
    end
    cd(PWD)

    ENV_ = copy(ENV)
    if Sys.iswindows()
        platform = "windows"
        ENV_["PATH"] = "$(homedir())/.julia/conda/3/Scripts;$(homedir())/.julia/conda/3/Library/bin;$(homedir())/.julia/conda/3/" * ENV_["PATH"]
    elseif Sys.islinux()
        if haskey(ENV, "GPU") && ENV["GPU"] in [1, "1"]
            platform = "linux-gpu"
        else
            platform = "linux"
        end
    else
        platform = "osx"
    end
    
    
    run(`$CONDA config --add channels conda-forge`)
    run(setenv(`$CONDA env update -n base --file $platform.yml`, ENV_))
    
end
