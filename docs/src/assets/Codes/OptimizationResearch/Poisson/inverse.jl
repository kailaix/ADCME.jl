using AdFem
using ADCME
using PyPlot 
using JLD2
using Revise
using ADCME 
using PyCall
DIR = abspath(joinpath(@__DIR__, ".."))
py"""
import sys
sys.path.insert(0, $DIR)
"""
opt = pyimport("optim.optim")


@load "data/fwd.jld2" SOL 

function f(x, y)
    return sin(2π*10y+π/8)
end

mmesh = Mesh(joinpath(PDATA, "twoholes_large.stl"))



using Random; Random.seed!(SEED)
idx = rand(1:length(SOL), length(SOL)÷5)
xy = gauss_nodes(mmesh)
θ = Variable(fc_init([2,20,20,20,1]))
Kappa = squeeze(fc(xy, [20, 20, 20,1], θ, activation="tanh")) + 2.0
F = eval_f_on_gauss_pts(f, mmesh)
L = compute_fem_laplace_matrix1(Kappa, mmesh)
RHS = compute_fem_source_term1(F, mmesh)

bd = bcnode(mmesh)
L, RHS = impose_Dirichlet_boundary_conditions(L, RHS, bd, zeros(length(bd)))

sol = L\RHS 
loss = sum((sol - SOL)^2)*1e10