using ADCME
using PyPlot


x0 = rand(100)
x0 = @. x0*0.4 + 0.3
x1 = collect(LinRange(0, 1, 100))
y0 = sin.(2π*x0)
w = Variable(fc_init([1, 20, 20, 20, 1]))
y = squeeze(fc(x0, [20, 20, 20, 1], w))
loss = sum((y - y0)^2)

sess = Session(); init(sess)
BFGS!(sess, loss)
y1 = run(sess, y)
plot(x0, y0, ".", label="Data")
x_dnn = run(sess,  squeeze(fc(x1, [20, 20, 20, 1], w)))
plot(x1, x_dnn,  "--", label="DNN Estimation")
legend()
w1 = run(sess, w)



##############################

μ = Variable(w1)
ρ = Variable(zeros(length(μ)))
σ = log(1+exp(ρ))

function likelihood(z)
    w = μ + σ * z
    y = squeeze(fc(x0, [20, 20, 20, 1], w))
    sum((y - y0)^2) - sum((w-μ)^2/(2σ^2)) + sum((w-w1)^2)
end

function inference(x)
    z = tf.random_normal((length(σ),), dtype=tf.float64)
    w = μ + σ * z
    y = squeeze(fc(x, [20, 20, 20, 1], w))|>squeeze
end

W = tf.random_normal((10, length(w)), dtype=tf.float64)
L = constant(0.0)
for i = 1:10
    global L += likelihood(W[i])
end

y2 = inference(x1)


opt = AdamOptimizer(0.01).minimize(L)
init(sess)
# run(sess, L)
losses = []
for i = 1:2000
    _, l = run(sess, [opt, L])
    push!(losses, l)
    @info i, l
end

Y = zeros(100, 1000)
for i = 1:1000
    Y[:,i] = run(sess, y2)
end

for i = 1:1000
    plot(x1, Y[:,i], "--", color="gray", alpha=0.5)
end
plot(x1, x_dnn, label="DNN Estimation")
plot(x0, y1, ".", label="Data")
legend()


##############################
# Naive Uncertainty Quantification 
function inference_naive(x)
    z = tf.random_normal((length(w1),), dtype=tf.float64)
    w = w1 + log(2)*z
    y = squeeze(fc(x, [20, 20, 20, 1], w))|>squeeze
end
y3 = inference(x1)

Y = zeros(100, 1000)
for i = 1:1000
    Y[:,i] = run(sess, y3)
end

for i = 1:1000
    plot(x1, Y[:,i], "--", color="gray", alpha=0.5)
end
plot(x1, x_dnn, label="DNN Estimation")
plot(x0, y1, ".", label="Data")
legend()
