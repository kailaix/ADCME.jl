## define variables...
INSTALL_GPU = Sys.islinux() && haskey(ENV, "GPU") && ENV["GPU"] in [1, "1"]

CONDA = ""
if Sys.iswindows()
    CONDA = "$(JULIA_ADCME_DIR)/.julia/adcme/Scripts/conda.exe"
else 
    CONDA = "$(JULIA_ADCME_DIR)/.julia/adcme/bin/conda"
end

INSTALLER = ""
if Sys.islinux() 
    INSTALLER = "Miniconda3-py37_4.8.3-Linux-x86_64.sh"
elseif Sys.isapple()
    INSTALLER = "Miniconda3-py37_4.8.3-MacOSX-x86_64.sh"
else 
    INSTALLER = "Miniconda3-py37_4.8.3-Windows-x86_64.exe"
end

function check_install()
    if Sys.iswindows()
    else
        for bin in ["unzip", "ninja"] 
            if !isfile(joinpath("$(JULIA_ADCME_DIR)/.julia/adcme/bin"), bin)
                return false 
            end
        end
    end
    return true
end

function install_conda()
    PWD = pwd()
    cd("$(JULIA_ADCME_DIR)/.julia/")
    if !(INSTALLER in readdir("."))
        @info "Downloading miniconda installer..."
        download("https://repo.anaconda.com/miniconda/"*INSTALLER, INSTALLER)
    end

    if isdir("adcme")
        if FORCE_REINSTALL_ADCME
            error("""ADCME dependencies directory already exist, and you indicate FORCE_REINSTALL_ADCME=true.
    Please (1) quit Julia, (2) remove the path $(joinpath(pwd(), "adcme")) and (3) rebuild ADCME.""")
        else 
            @info "ADCME dependencies have already been installed."
            return 
        end
    end

    @info "Installing miniconda..."
    if Sys.iswindows()
        run(`cmd /c start /wait "" $INSTALLER /InstallationType=JustMe /RegisterPython=0 /S /D=$(JULIA_ADCME_DIR)\\.julia\\adcme`)
    else
        run(`bash $INSTALLER -f -b -p adcme`)
    end
    cd(PWD)
end

function install_conda_envs()
    cd(@__DIR__)

    ENV_ = copy(ENV)
    if Sys.iswindows()
        platform = "windows"
        ENV_["PATH"] = "$(JULIA_ADCME_DIR)/.julia/adcme/Scripts;$(JULIA_ADCME_DIR)/.julia/adcme/Library/bin;$(JULIA_ADCME_DIR)/.julia/adcme/" * ENV_["PATH"]
    elseif Sys.islinux()
        platform = INSTALL_GPU ? "linux-gpu" : "linux"
    else
        platform = "osx"
    end
    if ((platform == "linux-gpu" && occursin("tensorflow-gpu", read(`$CONDA list`, String))) ||
        (platform in ["windows", "linux", "osx"] && occursin("tensorflow", read(`$CONDA list`, String)))) && 
        check_install()
        return 
    end 
    @info "Installing conda dependencies..."
    run(setenv(`$CONDA env update -n base --file $platform.yml`, ENV_))
end

install_conda()
install_conda_envs()
