# Managing Numerical Experiments with SMT


## Setup an SMT Workspace

Suppose we want to run many numerical , how do we manage our experiment and analysis?
ADCMEKit.jl provides a solution based on [Sumatra](https://github.com/open-research/sumatra).

When you install ADCMEKit.jl, a custom version of Sumatra is also installed on your system. Therefore,
you do not need to install anything manually.

Assume that we start a new project `MyProject` (folder name). We use Git for version control. To start a SMT server,
in the `MyProject` directory, do

```bash
smt init 
```

Now you can see a `.smt` directory in your folder.

Lets create some files. The first file is `test.jl`:

```julia
using DelimitedFiles
include("dep.jl")
writedlm("Data/data.txt", ones(10,10)*a)
```

The second file is `dep.jl`:

```julia
a = 10
```

One crucial step is to **commit** your files in the version control system because SMT manages files according to their git version.
After that, you can run

```bash
$ smt run -m test.jl
```

Here `-m` marks `test.jl` as the main file. We can also do a global configuration

```bash
$ smt configure -m test.jl
```

So next time we only need to call `smt run` to run the file.

SMT provides a web-based visualization tool to visualize your tasks. To start the web server, do

```bash
$ smtweb
```

Here are some snapshots for the web-based visualization tool.

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCMEKit/smt1.png?raw=true)
![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCMEKit/smt2.png?raw=true)

## Configurations


Typically when we want to run batch jobs using SMT, we can setup as follows:


**Creating an argument reading script: `readargs.jl`**

The `readargs.jl` file is responsible for parsing the input arguments. We explain by an example:

```julia
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--scaling"
            default = 0.002
            arg_type = Float64
        "--nsim"
            default = 5000
            arg_type = Int64
        "label"
            default = "nothing"
    end

    return parse_args(s)
end

parameters = parse_commandline()
println(parameters)

label = parameters["label"]
scaling = parameters["scaling"]
nsim = parameters["nsim"]
```

The lines that start with `--` in `@add_arg_table!` are keyword arguments. The lines that do not have hyphens are positional argument. By default, the last argument to the script is the label name. 

**Bash script**

An example of the bash script is as follows

```bash
set +x 
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0,1
smt configure -c store-diff
smt configure -d Data
smt configure --addlabel cmdline 
smt configure -e julia


for idx in 1 2 3
do  
for s in 0.0005 0.001 0.002 0.005 0.01
do 
smt run -m NNFWI-reg.jl --nsim 10 --scaling $s &
sleep 1
done 
done
wait 
```

Here we use `-c store-diff` so that we can make changes to the script without commits (not recommended). `-d Data` specifies the directory where SMT watches the file changes. `--addlabel cmdline` is mandatory if you want to run multiple jobs at the same time. This command ensures that SMT matches the output files with job labels. `-e julia` sets executable. 

