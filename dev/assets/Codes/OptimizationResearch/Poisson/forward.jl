

using AdFem
using ADCME
using PyPlot 
using JLD2

function kappa(x, y)
    return 2 + exp(10x) - (10y)^2
end

function f(x, y)
    return sin(2π*10y+π/8)
end

mmesh = Mesh(joinpath(PDATA, "twoholes_large.stl"))

Kappa = eval_f_on_gauss_pts(kappa, mmesh)
F = eval_f_on_gauss_pts(f, mmesh)
L = compute_fem_laplace_matrix1(Kappa, mmesh)
RHS = compute_fem_source_term1(F, mmesh)

bd = bcnode(mmesh)
L, RHS = impose_Dirichlet_boundary_conditions(L, RHS, bd, zeros(length(bd)))

SOL = L\RHS 
close("all")
figure(figsize = (10, 4))
subplot(121)
visualize_scalar_on_gauss_points(Kappa, mmesh)
title("\$\\kappa\$")
subplot(122)
visualize_scalar_on_fem_points(SOL, mmesh)
title("Solution")

make_directory("data")
savefig("data/fwd_ps.png")

@save "data/fwd.jld2" SOL