include("inverse.jl")

MODE = "ref"

# run(sess, loss)
loss_ = BFGS!(sess, loss, 100, vars = [E, nu], callback = cb)

E_, nu_ = run(sess, [E, nu])
close("all")
figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_gauss_points(E_, mmesh)
subplot(122)
visualize_scalar_on_gauss_points(nu_, mmesh)
savefig("data/final.png")
