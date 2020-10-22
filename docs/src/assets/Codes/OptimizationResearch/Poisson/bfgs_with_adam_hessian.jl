include("inverse.jl")
include("../optimizers.jl")

opt = AdamOptimizer().minimize(loss)
g = tf.convert_to_tensor(gradients(loss, θ))
sess = Session(); init(sess)

losses0 = Float64[]
B = diagm(0=>ones(length(θ)))
G, THETA = run(sess, [g,θ])

for i = 1:300
    _, l = run(sess, [opt, loss])
    G_, THETA_ = run(sess, [g,θ])

    global G, G_ = G_, G 
    global THETA, THETA_ = THETA_, THETA 
    s = THETA - THETA_
    y = G - G_ 
    global B = (I - s*y'/(y'*s)) * B * (I - y*s'/(y'*s)) + s*s'/(y'*s)

    @info i, l 
    push!(losses0, l)
end

# error()

losses = Optimize!(sess, loss; optimizer = BFGSOptimizer(), max_num_iter=700, B = B)

losses = [losses0;losses]
w = run(sess, θ)
@save "data/bfgs.jld2" losses w 

figure(figsize = (10, 4))
subplot(121)
semilogy(losses)
xlabel("Iterations"); ylabel("Loss")
subplot(122)
visualize_scalar_on_gauss_points(run(sess, Kappa), mmesh)
savefig("data/bfgs.png")