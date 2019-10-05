wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
sh -b -p Miniconda3-latest-Linux-x86_64.sh ~/.julia/conda/miniconda3
~/.julia/conda/miniconda3/bin/conda install --y tensorflow=1.14 tensorflow-probability=0.7 matplotlib
if [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    ~/.julia/conda/miniconda3/bin/conda install --y tensorflow-gpu=1.14
fi