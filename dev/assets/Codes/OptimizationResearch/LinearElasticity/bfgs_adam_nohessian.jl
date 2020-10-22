include("inverse.jl")

MODE = "bfgs_adam_nohessian"

make_directory("data/result$MODE")
opt = AdamOptimizer().minimize(loss)

include("../optimizers.jl")

sess = Session(); init(sess)
# run(sess, loss)
losses0 = Float64[]

for i = 1:300
    _, l = run(sess, [opt, loss])
    @info i, l 
    push!(losses0, l)
end


losses = Optimize!(sess, loss; vars = [θ1, θ2], optimizer = BFGSOptimizer(), max_num_iter=700)

losses = [losses0; losses]

THETA1, THETA2 = run(sess, [θ1, θ2])

@save "data/result$MODE/data.jld2" THETA1 THETA2 losses

close("all")
semilogy(losses)
savefig("data/result$MODE/loss.png")

E_, nu_ = run(sess, [E, nu])
close("all")
figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_gauss_points(E_, mmesh)
subplot(122)
visualize_scalar_on_gauss_points(nu_, mmesh)
savefig("data/result$MODE/final.png")
