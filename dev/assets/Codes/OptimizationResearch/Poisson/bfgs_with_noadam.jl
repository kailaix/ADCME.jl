include("inverse.jl")
include("../optimizers.jl")

opt = AdamOptimizer().minimize(loss)
g = tf.convert_to_tensor(gradients(loss, θ))
sess = Session(); init(sess)

losses0 = Float64[]

# error()

losses = Optimize!(sess, loss; optimizer = BFGSOptimizer(), max_num_iter=1000)

losses = [losses;losses0]
w = run(sess, θ)
@save "data/bfgs_noadam.jld2" losses w 

figure(figsize = (10, 4))
subplot(121)
semilogy(losses)
xlabel("Iterations"); ylabel("Loss")
subplot(122)
visualize_scalar_on_gauss_points(run(sess, Kappa), mmesh)
savefig("data/bfgs_noadam.png")