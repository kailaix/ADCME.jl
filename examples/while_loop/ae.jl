using ADCME
using PyPlot
encoding_dim = 32
x = placeholder(Float64, shape=[nothing, 784])
variable_scope("encoder", reuse=AUTO_REUSE) do
    net = x
    net = dense(net, 200, activation="relu")
    net = dense(net, 200, activation="relu")
    global encoded = dense(net, encoding_dim, activation="relu")
end
variable_scope("decoder", reuse=AUTO_REUSE) do
    net = dense(encoded, 200, activation="relu")
    net = dense(net, 200, activation="relu")
    global decoded = dense(net, 784, activation="sigmoid")
end

x_train, _, x_test, = mnist(Float64)
x_train /= 255
x_test /= 255
x_train = reshape(x_train, size(x_train,1), length(x_train[1,:,:]))
x_test = reshape(x_test, size(x_test,1), length(x_test[1,:,:]))
loss = sum(tf.keras.backend.binary_crossentropy(x, decoded))
variable_scope("nn2", reuse=AUTO_REUSE) do
    opt = AdamOptimizer()
    global train_op = minimize(opt, loss)
end
sess = Session()
init(sess)
batchsize=256
for i = 1:10000
    II = (1+(i-1)*batchsize):i*batchsize
    II = mod.(II, size(x_train,1)).+1
    data = x_train[II, :]
    _, los = run(sess, [train_op, loss], feed_dict=Dict(x=>data))
    if mod(i,100)==0
        println("#iter=$i, loss=$los")
    end
end

p = rand(1:size(x_train,1),6)
V = run(sess, decoded, feed_dict=Dict(x=>x_train[p,:]))
# V = Array{Int64}(V .> 0.5)
figure()
for i = 1:5
    subplot(2,5,i)
    imshow(reshape(V[i,:],28,28))
    subplot(2,5,i+5)
    imshow(reshape(x_train[p[i],:],28,28))
end
savefig("tmp.png")

ec = run(sess, encoded, feed_dict=Dict(x=>x_train[p,:]))
ec2 = (ec[1:3,:]+ec[4:6,:])/2

y = placeholder(Float64, shape=[nothing, 32])
variable_scope("decoder", reuse=AUTO_REUSE) do
    net = dense(y, 200, activation="relu")
    net = dense(net, 200, activation="relu")
    global imgs = dense(net, 784, activation="sigmoid")
end
IMGS = run(sess, imgs, feed_dict=Dict(y=>ec2))
# IMGS = IMGS .> 0.5
figure()
for i = 1:3
    subplot(1,3,i)
    imshow(reshape(IMGS[i,:],28,28))
end
savefig("tmp2.png")
