# How to install ADCME?

The simplest way to install ADCME is using the built-in Julia package manager. To do this, follow the steps below:

1. Install [Julia](https://julialang.org/)

2. Install `ADCME`
```julia
julia> ]
pkg> add ADCME
```
~~~julia
!!! note
To enable GPU support, first, make sure `nvcc` is available from your environment (e.g., 	type `nvcc` in your shell and you should get the location of the executable binary file). Second, rebuild ADCME with the following command
```julia
ENV["GPU"] = 1
Pkg.build("ADCME")
```
~~~
3. (Optional) Test `ADCME.jl`

```julia
julia> ]
pkg> test ADCME
```

## Troubleshooting







