using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using DelimitedFiles
using JLD2
Random.seed!(233)
ADCME.options.mpi.printlevel = 0

mutable struct MPIConfig 
    n::Int64 
    N::Int64
    h::Float64
    function MPIConfig(n::Int64)
        r = mpi_size()
        m = Int64(round(sqrt(r)))
        if m^2 != r 
            error("$r must be a squared number")
        end
        h = 1/(n*m+1)
        new(n, m, h)
    end
end

function getIJ(mc::MPIConfig)
    r = mpi_rank()
    N = mc.N 
    I = r÷N + 1
    J = r%N + 1 
    return I, J 
end

function get_xy(mc::MPIConfig)
    n, N, h = mc.n, mc.N, mc.h
    x = Float64[]
    y = Float64[]
    I, J = getIJ(mc)
    x0 = (I-1)*n*h 
    y0 = (J-1)*n *h 
    for i = 1:n 
        for j = 1:n 
            push!(x, i*h+x0)
            push!(y, j*h+y0)
        end
    end
    X, Y = Array(reshape(x, (mc.n, mc.n))'), Array(reshape(y, (mc.n, mc.n))')
end


"""
Given an extended `κ_ext` (size = `(n+2)×(n+2)`), solves the forward computation problem 
"""
function poisson_solver(κ_local, f_local, mc::MPIConfig)
    @assert size(κ_local) == (mc.n, mc.n)
    @assert size(f_local) == (mc.n, mc.n)
    global κ_ext = mpi_halo_exchange(κ_local, mc.N, mc.N; fill_value = 1.0)
    h = mc.h
    f_local = constant(f_local)
    rhs = 2*h^2*reshape(f_local, (-1,))

    poisson_linear_solver(κ_ext, rhs, mc)
end

function poisson_linear_solver(κ_ext, rhs, mc)
    function forward(κ_ext)
        A = get_poisson_matrix(κ_ext, mc)
        A\rhs 
    end

    function backward(du, u, κ_ext)
        A = get_poisson_matrix(κ_ext, mc)
        B = A'
        x = -(B\du)
        x = reshape(x, (mc.n, mc.n))
        u = reshape(u, (mc.n, mc.n))
        uext = mpi_halo_exchange(u, mc.N, mc.N; fill_value = 0.0)
        out = get_poisson_grad(x, uext, mc.N)
        set_shape(out, (mc.n+2, mc.n+2))
    end

    register(forward, backward)(κ_ext)
end


function get_poisson_grad(x,uext,N)
    get_poisson_grad_ = load_op_and_grad("$(@__DIR__)/build/libGetPoissonMatrix","get_poisson_grad")
    x,uext,cn = convert_to_tensor(Any[x,uext,N], [Float64,Float64,Int64])
    get_poisson_grad_(x,uext,cn)
end

function dofmap(mc::MPIConfig)
    n, N = mc.n, mc.N 
    global_to_local = zeros(Int64, (n*N)^2)
    local_to_global = zeros(Int64, (n*N)^2)
    for I = 1:N 
        for J = 1:N 
            for i = 1:n 
                for j = 1:n 
                    ii = (I-1)*n + i 
                    jj = (J-1)*n + j
                    global_to_local[ (ii-1)*n*N + jj ] = ((I-1)*N+J-1)*n^2 + (i-1)*n+j
                    local_to_global[ ((I-1)*N+J-1)*n^2 + (i-1)*n+j ] = (ii-1)*n*N + jj
                end
            end
        end
    end
    return global_to_local, local_to_global
end



function get_colext(mc::MPIConfig)
    I, J = getIJ(mc)
    n, N = mc.n, mc.N 
    NDOF = n*N 
    DMAP, _ = dofmap(mc)

    colext = zeros(Int64, n+2, n+2)
    for i = 1:n+2
        for j = 1:n+2
            ii = (I-1)*n+(i-1)
            jj = (J-1)*n+(j-1)
            if 1<=ii<=NDOF && 1<=jj<=NDOF
                idx = (ii-1)*NDOF + jj 
                colext[i,j] = DMAP[idx]
            end
        end
    end
    colext = colext .- 1

    ncolsize = Int64[]
    for i = 2:n+1
        for j = 2:n+1
            ns = (colext[i-1,j]>=0) + (colext[i+1,j]>=0) + 
                    (colext[i,j-1]>=0) + (colext[i,j+1]>=0) + (colext[i,j]>=0)
            push!(ncolsize, ns)
        end
    end

    ilower = ((I-1)*N + J-1)*n^2
    iupper = ((I-1)*N + J)*n^2-1
    
    rows = Int32.(Array(0:n^2 .- 1))
    rows = rows .+ ilower

    return colext, sum(ncolsize), Int32.(rows), Int32.(ncolsize), ilower, iupper
end

function get_poisson_matrix(kext,mc; deps = missing)
    colext, colssize, rows, ncols, ilower, iupper = get_colext(mc)
    if ismissing(deps)
        deps = kext[1,1]
    end
    get_poisson_matrix_ = load_op_and_grad("$(@__DIR__)/build/libGetPoissonMatrix","get_poisson_matrix", multiple=true)
    kext,colext,colssize,deps = convert_to_tensor(Any[kext,colext,colssize,deps], [Float64,Int64,Int64,Float64])
    global cols, vals = get_poisson_matrix_(kext,colext,colssize,deps)
    mpi_SparseTensor(rows, ncols, cols, vals, ilower, iupper, (mc.n*mc.N)^2)
end
