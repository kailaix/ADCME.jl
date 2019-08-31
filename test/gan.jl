using Revise
using ADCME
using PyPlot
matplotlib.use("macosx")
X,_,_,_ = mnist()
X = X/127.5 .-1.
Y = zeros(size(X,1), 28*28)
for i = 1:size(X,1)
    Y[i,:] = vec(X[i,:,:,1])
end
X = copy(Y)

hidden_units = [128,256]

function discriminator(x, gan)
    y = ae(x, [256,128,64,1])
    return y
end

function generator(x, gan)
    y = ae(x, [128,256,784])
    return y
end

function loss(t, f, gan)
    t_logits = discriminator(t, gan)
    f_logits = discriminator(f, gan)
    t_loss = mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=t_logits, labels=tf.ones_like(t_logits)))
    f_loss = mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
    d_loss = t_loss + f_loss
    g_loss = mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))
    d_loss, g_loss
end

reset_default_graph()
gan = GAN(X, generator, discriminator, loss; latent_dim=100, batch_size=32)
opt_d = RMSPropOptimizer(0.0002).minimize(gan.d_loss, var_list=gan.d_vars)
opt_g = RMSPropOptimizer(0.0002).minimize(gan.g_loss, var_list=gan.g_vars)
img = sample(gan, 25)
sess = Session(); init(sess)
# run(sess, gan.d_vars[1][1:10,1:10])
# run(sess, predict(gan, ones(100)))
rm("images/", force=true, recursive=true)
mkdir("images/")
for i = 1:200000
    feed = Dict(gan.noise=>randn(gan.batch_size, gan.latent_dim), gan.ids=>Int32.(rand(1:size(X,1), gan.batch_size)))
    _, dl = run(sess, [opt_d, gan.d_loss], feed_dict=feed)
    for i = 1:4
        run(sess, opt_g, feed_dict=feed)
    end
    _, _, gl = run(sess, [opt_g, gan.update, gan.g_loss], feed_dict=feed)
    if i%100==1
        println("iter=$i, dloss=$dl, gloss=$gl") 
    end
    if i%500==1
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