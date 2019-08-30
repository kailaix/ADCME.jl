using Revise
using ADCME
using PyPlot
matplotlib.use("macosx")
X,_,_,_ = mnist()
X = X/127.5 .-1.
X = reshape(X, size(X,1), :)

function discriminator(x, gan)
    local y
    y = dense(x,512)
    y = leaky_relu(y, alpha=0.2)
    y = dense(y,256)
    y = leaky_relu(y, alpha=0.2)
    y = dense(y, 1, activation="sigmoid")
    return y
end

function generator(x, gan)
    y = dense(x, 256)
    y = leaky_relu(y, alpha=0.2)
    # y = bn(y, momentum=0.8, is_training=gan.is_training)
    y = dense(y, 512)
    y = leaky_relu(y, alpha=0.2)
    # y = bn(y, momentum=0.8, is_training=gan.is_training)
    y = dense(y, 1024)
    y = leaky_relu(y, alpha=0.2)
    # y = bn(y, momentum=0.8, is_training=gan.is_training)
    y = dense(y, gan.dim, activation="tanh")
    return y
end

reset_default_graph()
gan = GAN(X, generator, discriminator, "jsgan"; latent_dim=100, batch_size=32)
opt_d = AdamOptimizer(0.0002, beta1=0.5).minimize(gan.d_loss, var_list=gan.d_vars)
opt_g = AdamOptimizer(0.0002, beta1=0.5).minimize(gan.g_loss, var_list=gan.g_vars)
img = sample(gan, 25)
sess = Session(); init(sess)
rm("images/", force=true, recursive=true)
mkdir("images/")
for i = 1:30000
    _, dl = run(sess, [opt_d, gan.d_loss])
    _, _, gl = run(sess, [opt_g, gan.update, gan.g_loss])
    println("iter=$i, dloss=$dl, gloss=$gl")
    if i%200==0
        val = run(sess, img)
        fig, axs = subplots(5,5)
        cnt = 1
        for i = 1:5
            for j = 1:5
                axs[i,j].imshow(reshape(val[cnt,:], 28, 28), cmap="gray")
                axs[i,j].axis("off")
                cnt += 1
            end
        end
        savefig("images/$i.png")
        close("all")
    end
end