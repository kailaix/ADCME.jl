@testset "sinkhorn" begin
    a = Float64.(1:5)/sum(1:5)
    b = Float64.(1:10)/sum(1:10)
    m = zeros(5, 10)
    for i = 1:5
        for j = 1:10
            m[i,j] = 1/(i+j)
        end
    end
    reg = 1.0
    iter = 100
    tol = 1e-10
    loss = sinkhorn(a, b, m)
    @test run(sess, loss) ≈ 0.10269121
    # @test_nowarn loss = sinkhorn(a, b, m, method = "lp")
end

@testset "dist" begin
    a = rand(10,3)
    b = rand(20,3)
    for order in [1,2,3,4,5]
        M = zeros(10, 20)
        for i = 1:10
            for j = 1:20
                M[i,j] = norm(a[i,:] - b[j,:], order)
            end
        end
        m = ot_dist(a, b, order)
        @test m≈M 
        m = ot_dist(a, constant(b), order)
        @test run(sess, m)≈M
    end
end

# @testset "fastdtw" begin 
#     Sample = Float64[1,2,3,5,5,5,6]
#     Test = Float64[1,1,2,2,3,5]
#     u, p = dtw(Sample, Test, true)
#     @test run(sess, u)≈1.0
# end
