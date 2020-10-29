include("inverse.jl")
include("../optimizers.jl")

N = 300

opt = AdamOptimizer().minimize(loss)
g = tf.convert_to_tensor(gradients(loss, θ))
sess = Session(); init(sess)

losses0 = Float64[]
B = diagm(0=>ones(length(θ)))
G, THETA = run(sess, [g,θ])

for i = 1:N
    _, l = run(sess, [opt, loss])
    @info i, l 
    push!(losses0, l)
end

# error()

losses = Optimize!(sess, loss; optimizer = BFGSOptimizer(), max_num_iter=1000-N, β=0.8)

losses = [losses0;losses;]
w = run(sess, θ)
@save "data/damped_bfgs_with_adam_nohessian$SEED.jld2" losses w 

figure(figsize = (10, 4))
subplot(121)
semilogy(losses)
xlabel("Iterations"); ylabel("Loss")
subplot(122)
visualize_scalar_on_gauss_points(run(sess, Kappa), mmesh)
savefig("data/damped_bfgs_with_adam_nohessian$SEED.png")