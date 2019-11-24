# adapted from https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN

using ADCME
using PyCall
using PyPlot
using DelimitedFiles

tt = pyimport("tensorflow.examples.tutorials.mnist")

function lrelu(x, th=0.2)
    return tf.maximum(th * x, x)
end

function generator(x, isTrain=true)
    local o
    variable_scope("generator") do

        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(x, 1024, [4, 4], strides=(1, 1), padding="valid")
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding="same")
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding="same")
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding="same")
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output layer
        conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding="same")
        o = tf.nn.tanh(conv5)
    end
    return o
end

function discriminator(x, isTrain=true)
    local o, conv5
    variable_scope("discriminator") do
        # 1st hidden layer
        conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding="same")
        lrelu1 = lrelu(conv1, 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding="same")
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding="same")
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding="same")
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output layer
        conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding="valid")
        o = tf.nn.sigmoid(conv5)

        
    end
    return o, conv5
end

let fixed_z_ = randn(25, 1, 1, 100)
    global show_result
    function show_result(filename)
        test_images = run(sess, G_z, feed_dict=Dict(z=> fixed_z_, isTrain=>false))
    
        size_figure_grid = 5
        fig, ax = subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i = 1:5
            for j = 1:5
                ax[i, j].get_xaxis().set_visible(false)
                ax[i, j].get_yaxis().set_visible(false)
            end
        end
    
        for k = 1:size_figure_grid*size_figure_grid
            i = div(k-1, size_figure_grid)+1
            j = (k%size_figure_grid) + 1
            ax[i, j].cla()
            ax[i, j].imshow(reshape(test_images[k,:,:,:], 64, 64), cmap="gray")
        end
    
        label = "Epoch $filename"
        fig.text(0.5, 0.04, label, ha="center")
    
        savefig(filename)
        close("all")
    end
end

# training parameters
batch_size = 100
lr = 0.0002
train_epoch = 20

if !isdir("MNIST_data")
    mkdir("MNIST_data")
    download("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "MNIST_data/train-images-idx3-ubyte.gz")
    download("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "MNIST_data/t10k-labels-idx1-ubyte.gz")
    download("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "MNIST_data/t10k-images-idx3-ubyte.gz")
    download("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "MNIST_data/train-labels-idx1-ubyte.gz")
end
if !isdir("figures")
    mkdir("figures")
end

mnist = tt.input_data.read_data_sets("MNIST_data/", one_hot=true, reshape=[])


x = placeholder(Float32, shape=(nothing, 64, 64, 1))
z = placeholder(Float32, shape=(nothing, 1, 1, 100))
isTrain = placeholder(Bool)

G_z = generator(z, isTrain)
# networks : discriminator
D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain)

# loss for each network
D_loss_real = mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))


D_vars = [x for x in get_collection() if startswith(x.name, "discriminator")]
G_vars = [x for x in get_collection() if startswith(x.name, "generator")]

control_dependencies(get_collection(tf.GraphKeys.UPDATE_OPS)) do 
    global D_optim = AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    global G_optim = AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)
end

sess = Session(); init(sess)
train_set = run(sess, tf.image.resize_images(mnist[1].images, [64, 64]))
train_set = (train_set - 0.5) / 0.5
num_examples = size(train_set, 1)

loss_d = []
loss_g = []
for epoch = 1:train_epoch
    for iter = 1:div(num_examples, batch_size)
        x_ = train_set[(iter-1)*batch_size+1:iter*batch_size,:,:,:]
        z_ = randn(batch_size, 1, 1, 100)
        loss_d_, _ = run(sess, [D_loss, D_optim], feed_dict=Dict(x=>x_, z=>z_, isTrain=>true))

        z_ = randn(batch_size, 1, 1, 100)
        loss_g_, _ = run(sess, [G_loss, G_optim], feed_dict=Dict(x=>x_, z=>z_, isTrain=>true))
        
        println("[$epoch, #$iter] $loss_d_, $loss_g_")
        push!(loss_d, loss_d_)
        push!(loss_g, loss_g_)
        show_result("figures/$(epoch)_$(iter)")
    end

end

close("all")
plot(loss_d, label="Discriminator")
plot(loss_g, label="Generator")
xlabel("Iterations")
ylabel("Loss")
savefig("figures/loss.png")

writedlm("figures/lossd.txt", loss_d)
writedlm("figures/lossg.txt", loss_g)