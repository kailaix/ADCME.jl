using Revise
using ADCME
using NNFEM
using LinearAlgebra
using PyPlot
using JLD2 
using Statistics
include("compute_loss.jl")
sess = Session(); init(sess)

@info run(sess, loss)
BFGS!(sess, loss, 100)

# figure()
# visualize_mesh(domain)
# scatter(domain.nodes[idx,1], domain.nodes[idx,2], color="r", label="Observation")
# legend()


# k_ = run(sess, k0)
# sol_ = run(sess, s)

# figure(figsize=(10, 5))
# subplot(121)
# title("\$\\kappa(x,y)\$")
# visualize_scalar_on_undeformed_body(k_, domain)
# subplot(122)
# visualize_scalar_on_undeformed_body(sol_, domain)
# title("\$u(x,y)\$")
