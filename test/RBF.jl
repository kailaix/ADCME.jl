@testset "kMeans" begin
    using Random; Random.seed!(233)
    data = rand(100,2)
    ic = kMeansInitCentroids(data, 10)
    centroids, membership = kMeans(data, ic, 100)
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