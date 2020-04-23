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