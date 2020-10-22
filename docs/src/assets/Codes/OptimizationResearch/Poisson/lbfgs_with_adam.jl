include("inverse.jl")
include("../optimizers.jl")

N = 300
if length(ARGS)==1
    global N = parse(Int64, ARGS[1])
    @info "N=$N"
end

opt = AdamOptimizer().minimize(loss)
g = tf.convert_to_tensor(gradients(loss, θ))
sess = Session(); init(sess)

losses0 = Float64[]

for i = 1:N
    _, l = run(sess, [opt, loss])
    @info i, l 
    push!(losses0, l)
end

# error()

losses = Optimize!(sess, loss; optimizer = LBFGSOptimizer(), max_num_iter=1000-N, m = 50)

losses = [losses0;losses]
w = run(sess, θ)
@save "data/lbfgs_adam$N.jld2" losses w 

figure(figsize = (10, 4))
subplot(121)
semilogy(losses)
xlabel("Iterations"); ylabel("Loss")
subplot(122)
visualize_scalar_on_gauss_points(run(sess, Kappa), mmesh)
savefig("data/lbfgs_adam$N.png")