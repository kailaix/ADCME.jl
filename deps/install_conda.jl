installer = ""
if Sys.islinux() 
    installer = "Miniconda3-py37_4.8.3-Linux-x86_64.sh"
elseif Sys.isapple()
    installer = "Miniconda3-py37_4.8.3-MacOSX-x86_64.sh"
else 
    installer = "Miniconda3-py37_4.8.3-Windows-x86_64.exe"
end

if isdir("$(homedir())/.julia/conda/")
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