# example from https://github.com/malmaud/TensorFlow.jl
using Distributions
using Printf

# Generate some synthetic data
x = randn(100, 50)
w = randn(50, 10)
y_prob = exp.(x*w)
y_prob ./= sum(y_prob,dims=2)

function draw(probs)
    y = zeros(size(probs))
    for i in 1:size(probs, 1)
        idx = rand(Categorical(probs[i, :]))
        y[i, idx] = 1
    end
    return y
end

y = draw(y_prob)

# Build the model
reset_default_graph()
sess = Session()

X = placeholder(Float64, shape=[nothing, 50])
Y_obs = placeholder(Float64, shape=[nothing, 10])

variable_scope("test_logisitic_model"; initializer=random_normal_initializer(0, .001), reuse=AUTO_REUSE) do
    global W = get_variable("W", shape=[50, 10], dtype=Float64)
    global B = get_variable("B", shape=[10], dtype=Float64)
end

Y=softmax(X*W + B)

Loss = -sum(log(Y).*Y_obs)
# adam = AdamOptimizer()
# minimize_op = minimize(adam, Loss)

# # Run training
# run(sess, global_variables_initializer())
# for epoch in 1:1000
#     cur_loss, _ = run(sess, [Loss, minimize_op], feed_dict=Dict(X=>x, Y_obs=>y))
#     println(@sprintf("Current loss is %.2f.", cur_loss))
# end

function print_loss(loss_evaled)
    println(loss_evaled)
end

opt = ScipyOptimizerInterface(Loss, method="L-BFGS-B",options=Dict("maxiter"=> 100))
run(sess, global_variables_initializer())
ScipyOptimizerMinimize(sess, opt, feed_dict=Dict(X=>x, Y_obs=>y), loss_callback=print_loss, fetches=[Loss])
