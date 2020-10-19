if Sys.islinux() && haskey(ENV, "GPU") 
    try 
        run(`which nvcc`)
    catch 
        error("You specified GPU=1 but NVCC is not available. Please include `nvcc` in your system path.")
    end
end

INSTALL_GPU = Sys.islinux() && haskey(ENV, "GPU") 

CONDA = ""
if Sys.iswindows()
    CONDA = "$(JULIA_ADCME_DIR)/.julia/adcme/Scripts/conda.exe"
else 
    CONDA = "$(JULIA_ADCME_DIR)/.julia/adcme/bin/conda"
end


if (!FORCE_REINSTALL_ADCME) && isfile(CONDA) && 
    (occursin("tensorflow-gpu", read(`$CONDA list`, String)) || 
        (!INSTALL_GPU && occursin("tensorflow", read(`$CONDA list`, String))))
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

    PWD = pwd()
    cd("$(JULIA_ADCME_DIR)/.julia/")
    if !(installer in readdir("."))
        @info "Downloading miniconda installer..."
        download("https://repo.anaconda.com/miniconda/"*installer, installer)
    end

    if isdir("adcme") && FORCE_REINSTALL_ADCME
        error("""$(joinpath(pwd(), "adcme")) already exists. Please quit Julia, remove the path and try again.""")
    end

    @info "Installing miniconda..."
    if Sys.iswindows()
        run(`cmd /c start /wait "" $installer /InstallationType=JustMe /RegisterPython=0 /S /D=$(JULIA_ADCME_DIR)\\.julia\\adcme`)
    else
        run(`bash $installer -f -b -p adcme`)
    end
    cd(PWD)

    ENV_ = copy(ENV)
    if Sys.iswindows()
        platform = "windows"
        ENV_["PATH"] = "$(JULIA_ADCME_DIR)/.julia/adcme/Scripts;$(JULIA_ADCME_DIR)/.julia/adcme/Library/bin;$(JULIA_ADCME_DIR)/.julia/adcme/" * ENV_["PATH"]
    elseif Sys.islinux()
        if haskey(ENV, "GPU") && ENV["GPU"] in [1, "1"]
            platform = "linux-gpu"
        else
            platform = "linux"
        end
    else
        platform = "osx"
    end
    
    run(setenv(`$CONDA env update -n base --file $platform.yml`, ENV_))
    
end
