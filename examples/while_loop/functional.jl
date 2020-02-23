using ADCME 
using PyPlot

function residual_and_jacobian(θ, u)
    X = ae(u, config, θ) + 1.0     # (1)
    Xp = tf.gradients(X, u)[1]
    Xpp = tf.gradients(Xp, u)[1]
    up = [u[2:end];constant(zeros(1))]
    un = [constant(zeros(1)); u[1:end-1]]
    R = Xp .* ((up-un)/2h)^2 + X .* (up+un-2u)/h^2 - φ
    dRdu = Xpp .* ((up-un)/2h)^2 + Xp.*(up+un-2u)/h^2 - 2/h^2*X 
    dRdun = -Xp[2:end]/h .* (up-un)[2:end]/2h + X[2:end]/h^2
    dRdup = Xp[1:end-1]/h .* (up-un)[1:end-1]/2h + X[1:end-1]/h^2
    J = spdiag(n-1, 
        -1=>dRdun,
        0=>dRdu,
        1=>dRdup)     # (2)
    return R, J
end


config = [20,20,20,1]
n = 100
h = 1/n
x = collect(LinRange(0, 1.0, n+1))

φ = @. (1 - 2*x)*(-100*x^2*(2*x - 2) - 200*x*(1 - x)^2)/(100*x^2*(1 - x)^2 + 1)^2 - 2 - 2/(100*x^2*(1 - x)^2 + 1)
φ = φ[2:end-1]
θ = Variable(ae_init([1,config...]))
u0 = constant(zeros(n-1)) 
function L(u)    # (3)
  u_obs = (@. x * (1-x))[2:end-1]
  loss = mean((u - u_obs)^2) 
end
loss, solution, grad = NonlinearConstrainedProblem(residual_and_jacobian, L, θ, u0)
X_pred = ae(collect(LinRange(0.0,0.25,100)), config, θ) + 1.0

sess = Session(); init(sess)
BFGS!(sess, loss, grad, θ)
x_pred, sol = run(sess, [X_pred, solution])

figure(figsize=(10,4))
subplot(121)
s = LinRange(0.0,0.25,100)
x_exact = @. 1/(1+100*s^2) + 1
plot(s, x_exact, "-", linewidth=3, label="Exact")
plot(s, x_pred, "o", markersize=2, label="Estimated")
legend()
xlabel("u")
ylabel("X(u)")

subplot(122)
s = LinRange(0.0,1.0,101)[2:end-1]
plot(s, (@. s * (1-s)), "-", linewidth=3, label="Exact")
plot(s, sol, "o", markersize=2, label="Estimated")
legend()
xlabel("x")
ylabel("u")
savefig("nn.png")