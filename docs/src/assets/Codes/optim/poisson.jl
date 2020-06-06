using Revise
using ADCME
using NNFEM
using LinearAlgebra
using PyPlot
using JLD2 

nodes, elems = meshread("$(splitdir(pathof(NNFEM))[1])/../deps/Data/lshape_dense.msh")
elements = []
prop = Dict("name"=> "Scalar1D", "kappa"=>2.0)

for k = 1:size(elems,1)
    elnodes = elems[k,:]
    ngp = 2
    coord = nodes[elnodes,:]
    push!(elements, SmallStrainContinuum(coord, elnodes, prop, ngp))
end


EBC = zeros(Int64, size(nodes,1))
FBC = zeros(Int64, size(nodes,1))
g = zeros(size(nodes,1))
f = zeros(size(nodes,1))

bd = find_boundary(nodes, elems)
EBC[bd] .= -1

ndims = 1
domain = Domain(nodes, elements, ndims, EBC, g, FBC, f)
init_nnfem(domain)

α = 0.4*π/2
d = [cos(α);sin(α)]
f = (x,y)->sin(2π*y + π/8)
fext = compute_body_force_terms1(domain, f)

sol = zeros(domain.nnodes)
gnodes = getGaussPoints(domain)
x, y = gnodes[:,1], gnodes[:,2]

kref = @. 2.0 + exp(x) - y^2
k = vector(1:4:4getNGauss(domain), kref, 4getNGauss(domain)) + vector(4:4:4getNGauss(domain), kref, 4getNGauss(domain))
k = reshape(k, (getNGauss(domain),2,2))
K = s_compute_stiffness_matrix1(k, domain)
S = K\fext

sess = Session(); init(sess)
sol[domain.dof_to_eq] = run(sess, S)

figure(figsize=(10, 5))
subplot(121)
title("\$\\kappa(x,y)\$")
visualize_scalar_on_undeformed_body(kref, domain)
subplot(122)
visualize_scalar_on_undeformed_body(sol, domain)
title("\$u(x,y)\$")

@save "poisson.jld2" domain kref sol 
