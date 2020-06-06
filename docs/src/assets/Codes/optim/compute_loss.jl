
reset_default_graph() 

@load "poisson.jld2" domain kref sol 
init_nnfem(domain)
α = 0.4*π/2
d = [cos(α);sin(α)]
f = (x,y)->sin(2π*y + π/8)
fext = compute_body_force_terms1(domain, f)

gnodes = getGaussPoints(domain)
x, y = gnodes[:,1], gnodes[:,2]

k0 = fc([x y], [20,20,20,1])|>squeeze
k0 = softplus(k0)

k = vector(1:4:4getNGauss(domain), k0, 4getNGauss(domain)) + vector(4:4:4getNGauss(domain), k0, 4getNGauss(domain))
k = reshape(k, (getNGauss(domain),2,2))
K = s_compute_stiffness_matrix1(k, domain)
S = K\fext
s = vector(findall(domain.dof_to_eq), S, domain.nnodes)
using Random; Random.seed!(233)
idx = rand(findall(domain.dof_to_eq), 20)
loss = mean((s[idx] - sol[idx])^2)