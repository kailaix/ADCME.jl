# This is an example of WGAN_GP using ADCME.jl
# Reference: https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py#L147
using ADCME
using PyPlot
# step 0: parameters
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100
variable_scope("rms", reuse=AUTO_REUSE) do
    global optimizer1 = RMSPropOptimizer(0.00005)
    global optimizer2 = RMSPropOptimizer(0.00005)
end
n_critic = 5
batch_size = 32
Xtrain,_,_,_ = mnist(Float32)
Xtrain = (Xtrain.-127.5f0)/127.5f0 
Xtrain = reshape(Xtrain, size(Xtrain)...,1)

function generator(x)
    scope = "generator"
    variable_scope(scope, reuse=AUTO_REUSE) do
        x = dense(x, 128*7*7, activation="relu")
        x = Reshape(x, (7,7,128),name="2")
        x = UpSampling2D(x,name="3")
        x = conv2d(x, 128, kernel_size=4, padding="same")
        # x = BatchNormalization(x, momentum=0.8)
        x = relu(x)
        x = UpSampling2D(x,name="5")
        x = conv2d(x, 64, kernel_size=4, padding="same")
        # x = BatchNormalization(x,momentum=0.8)
        x = relu(x)
        x = conv2d(x, channels, kernel_size=4, padding="same")
        x = tanh(x)
    end
    return x
end

function critic(x)
    variable_scope("critic", reuse=AUTO_REUSE) do
        x = conv2d(x, 16, kernel_size=3, strides=2, padding="same")
        x = leaky_relu(x, alpha=0.2)
        x = dropout(x, 0.25)
        x = conv2d(x, 32, kernel_size=3, strides=2, padding="same")
        x = ZeroPadding2D(x, padding=((0,1),(0,1)))
        # x = BatchNormalization(x, momentum=0.8)
        x = leaky_relu(x, alpha=0.2)
        x = dropout(x, 0.25)
        x = conv2d(x, 64, kernel_size=3, strides=2, padding="same")
        # x = BatchNormalization(x, momentum=0.8)
        x = leaky_relu(x, alpha=0.2)
        x = dropout(x, 0.25)
        x = conv2d(x, 128, kernel_size=3, strides=1, padding="same")
        # x = BatchNormalization(x, momentum=0.8)
        x = leaky_relu(x, alpha=0.2)
        x = dropout(x, 0.25)
        x = flatten(x)
        x = dense(x, 1)
        x = sigmoid(x)
    end
    return x
end


function wasserstein_loss(ypred, ytrue)
    return mean(ytrue.*ypred)
end

function gradient_penalty_loss(y_pred, averaged_samples)
    grad = tf.gradients(y_pred, averaged_samples)[1]
    grd_sqr_sum = sum(grad^2, dims=[2;3;4])
    grd_l2_norm = sqrt(grd_sqr_sum)
    gradient_penalty = (1 - grd_l2_norm)^2
    return mean(gradient_penalty)
end

# step 1: create generator neural network
real_img = placeholder(Float32, shape=[batch_size, img_shape...])
z_disc = placeholder(Float32, shape=[batch_size, latent_dim])
fake_img = generator(z_disc)
# error("inspection")
fake = critic(fake_img)
valid = critic(real_img)

α = random_uniform((32,1,1,1), dtype=Float32)
interpolated_img = α * real_img + (1 - α) * fake_img
validity_interpolated = critic(interpolated_img)

l1 = wasserstein_loss(valid, -ones(Float32,batch_size,1))
l2 = wasserstein_loss(fake, ones(Float32,batch_size,1))
l3 = gradient_penalty_loss(validity_interpolated, interpolated_img)
l_critic = l1 + l2 + 10*l3

z_gen = placeholder(Float32, shape=[batch_size, latent_dim])
img = generator(z_gen)
valid = critic(img)
l_generator = wasserstein_loss(valid, -ones(Float32,batch_size,1))

# error("inspection")
critic_var = get_collection("critic")
generator_var = get_collection("generator")
variable_scope("rms", reuse=AUTO_REUSE) do
    global train_critic = minimize(optimizer1, l_critic, var_list=critic_var)
    global train_generator = minimize(optimizer2, l_generator, var_list=generator_var)
end


si_z_gen = placeholder(Float32, shape=[25, 100])
si_img = generator(si_z_gen)
function sample_images(epoch)
    noise = randn(25, latent_dim)
    gen_images = run(sess, si_img, feed_dict=Dict(si_z_gen=>noise))
    gen_images = 0.5*gen_images .+ 1
    close("all")
    figure()
    cnt = 1
    for i = 1:5
        for j = 1:5
            subplot(5,5,i+(j-1)*5)
            imshow(gen_images[cnt,:,:,1], cmap="gray")
            axis("off")
            cnt+=1
        end
    end
    savefig("images/mnist_$epoch.png")
    close("all")
end

# # step 2: training
sess = Session()
init(sess)
epochs = 600000
d_loss = nothing
noise  = nothing
_l1    = nothing
_l2    = nothing
_l3    = nothing
noise  = nothing
for epoch = 1:epochs
    for _ = 1:n_critic
        global d_loss, _l1, _l2, _l3, noise
        idx = rand(1:size(Xtrain,1), batch_size)
        imgs = Xtrain[idx,:,:,:]
        noise = randn(Float32, batch_size, latent_dim)
        _l1,_l2,_l3,d_loss,_ = run(sess, [l1,l2,l3,l_critic, train_critic], 
            feed_dict=Dict(real_img=>imgs, z_disc=>noise))
    end
    # noise = randn(batch_size, latent_dim)
    #k1,k2 = run(sess, [l2, l_generator], feed_dict=Dict(z_disc=>noise, z_gen=>noise))
    #@show k1, k2
    g_loss,_ = run(sess, [l_generator, train_generator], 
            feed_dict=Dict(z_gen=>noise))
    if mod(epoch, 100)==1
        sample_images(epoch)
    end
    println("$epoch [D loss: $(d_loss)] [G loss: $(g_loss)]")
    println("Diagnose: $_l1, $_l2, $_l3")
end
