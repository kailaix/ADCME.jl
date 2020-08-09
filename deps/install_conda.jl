installer = ""
if Sys.islinux()
    installer = "Miniconda3-py37_4.8.3-Linux-x86_64.sh"
elseif Sys.isapple()
    installer = "Miniconda3-py37_4.8.3-MacOSX-x86_64.sh"
else 
    error("Currently installing miniconda on Windows does not work.")
end

if isdir("~/.julia/conda/")
    mkpath("~/.julia/conda/")
end 

PWD = pwd()
cd("~/.julia/conda/")
@info "Downloading miniconda installer..."
download("https://repo.anaconda.com/miniconda/"*installer, installer)
if isdir("3")
    mv("3", "trash")
end
@info "Installing miniconda..."
run(`bash $installer -b -p 3`)
cd(PWD)