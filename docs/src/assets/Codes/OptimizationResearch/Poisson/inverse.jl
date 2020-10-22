using AdFem
using ADCME
using PyPlot 
using JLD2

@load "data/fwd.jld2" SOL 

function f(x, y)
    return sin(2π*10y+π/8)
end

mmesh = Mesh(joinpath(PDATA, "twoholes_large.stl"))

using Random; Random.seed!(233)
idx = rand(1:length(SOL), length(SOL)÷5)
xy = gauss_nodes(mmesh)
θ = Variable(fc_init([2,20,20,20,1]))
Kappa = squeeze(fc(xy, [20, 20, 20, 1], θ)) + 1.0
F = eval_f_on_gauss_pts(f, mmesh)
L = compute_fem_laplace_matrix1(Kappa, mmesh)
RHS = compute_fem_source_term1(F, mmesh)

bd = bcnode(mmesh)
L, RHS = impose_Dirichlet_boundary_conditions(L, RHS, bd, zeros(length(bd)))

sol = L\RHS 
loss = sum((sol - SOL)^2)*1e20