reset_default_graph(); sess = Session()
@testset "indexing for rank 3 tensors" begin 
    a = rand(100,10,20)
    i1 = 3:8
    i2 = [1;3;4]
    V = a[i1,i2,:]
    P = constant(a)[i1,i2,:]
    @test run(sess, P)≈V
    P = constant(a)[2,i2,:]
    @test run(sess, P)≈a[2,i2,:]
    P = constant(a)[2,i2,5]
    @test run(sess, P)≈a[2,i2,5]
end

@testset "fcx" begin 
    config = [2, 20,20,20,3]
    x = constant(rand(10,2))
    θ = ae_init(config)
    u, du = fcx(x, config[2:end], θ)
    y = ae(x, config[2:end], θ)

    @test run(sess, u) ≈ run(sess, y)

    DU = run(sess, du)
    @test run(sess, tf.gradients(y[:,1], x)[1]) ≈ DU[:,1,:]
    @test run(sess, tf.gradients(y[:,2], x)[1]) ≈ DU[:,2,:]
    @test run(sess, tf.gradients(y[:,3], x)[1]) ≈ DU[:,3,:]
end

@testset "dropout" begin 
    a = rand(10,2)
    @test sum(run(sess, dropout(a, 0.999, true)).==0)/20>0.9
    @test sum(run(sess, dropout(a, 0.999, false)).==0)==0
end