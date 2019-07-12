@testset "kMeans" begin
    using Random; Random.seed!(233)
    data = rand(100,2)
    centroids, membership = kMeans(data, 10, 100)
    @test norm(centroids-[0.121413  0.182218
    0.840211  0.138995
    0.881158  0.483758
    0.120535  0.48452 
    0.101097  0.859932
    0.459625  0.397973
    0.726131  0.95333 
    0.378898  0.791251
    0.766469  0.747534
    0.604228  0.702274])<1e-5
end

@testset "RBF" begin
    using Random; Random.seed!(233)
    X =  2*(rand(100,2).-0.5)
    Y = X[:,1].^2+X[:,2].^2

    k = 10
    centroids, membership = kMeans(X, k, 100)
    β = Variable(computeRBFBetas(X,centroids, membership))
    centroids = Variable(centroids)
    θ = Variable(ones(k+1))
    x = constant(X)
    y = evaluateRBFN(centroids, β, θ, x, true)
    # loss = sum((Y-y)^2)
    # opt = ScipyOptimizerInterface(loss, options=Dict("maxiter"=> 50))
    # init(sess)
    # ScipyOptimizerMinimize(sess,opt)
    # Y0 = run(sess,y)
    # close("all")
    # scatter3D(X[:,1],X[:,2],Y0)
    # scatter3D(X[:,1],X[:,2],Y)
    # xlim([-0.5,0.5])
    # ylim([-0.5,0.5])
    # run(sess, y)
end