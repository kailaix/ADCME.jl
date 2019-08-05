@testset "stddev" begin
    μ = rand(10)
    B = rand(10,10)
    C = B*B'; c = sqrt.(diag(C))
    u = UQNode(μ, C)
    @test run(sess, stddev(u)) ≈ c
end

@testset "ml_estimator" begin
    X = rand(20,5)
    C = cov(X); μ = mean(X, dims=1)[:]
    m, c = ml_estimator(X)
    res = run(sess, [m,c])
    @test res[1]≈μ
    @test res[2]≈C
    m, c = ml_estimator(constant(X))
    res = run(sess, [m,c])
    @test res[1]≈μ
    @test res[2]≈C
end

@testset "arithmetic" begin
    μ1 = rand(10); μ2 = rand(10)
    C1 = rand(10); C2 = rand(10); C1 = C1*C1'; C2 = C2*C2'
    u1 = UQNode(μ1, C1); u2 = UQNode(μ2, C2)
    u = u1 + u2
    @test run(sess, u.cov)≈C1+C2
    @test run(sess, u.loc)≈μ1+μ2

    u = u1 - u2
    @test run(sess, u.cov)≈C1+C2
    @test run(sess, u.loc)≈μ1-μ2
    
    u = -u1
    @test run(sess, u.cov)≈C1
    @test run(sess, u.loc)≈-μ1
    
    u = u1 + ones(10)
    @test run(sess, u.cov)≈C1
    @test run(sess, u.loc)≈μ1 .+ 1

    u =  ones(10) + u1
    @test run(sess, u.cov)≈C1
    @test run(sess, u.loc)≈μ1 .+ 1
    
    u = u1 + 1.0
    @test run(sess, u.cov)≈C1
    @test run(sess, u.loc)≈μ1 .+ 1.0

    u = 1.0 + u1 
    @test run(sess, u.cov)≈C1
    @test run(sess, u.loc)≈μ1 .+ 1.0


    u = u1 * 2.0
    @test run(sess, u.cov)≈C1 * 4
    @test run(sess, u.loc)≈μ1 * 2.0

    u = u1 / 2.0
    @test run(sess, u.cov)≈C1 / 4.0
    @test run(sess, u.loc)≈μ1 / 2.0

    u = 2.0 * u1
    @test run(sess, u.cov)≈C1 * 4
    @test run(sess, u.loc)≈μ1 * 2.0


    G = rand(10,10)
    u = G*u1
    @test run(sess, u.cov)≈ G * C1 * G'
    @test run(sess, u.loc)≈ G * μ1

    g = rand(10)
    u = g.*u1
    @test run(sess, u.cov)≈ diagm(0=>g) * C1 * diagm(0=>g)
    @test run(sess, u.loc)≈μ1 .* g

    g = rand(10)
    u = u1 ./ g
    @test run(sess, u.cov)≈ diagm(0=>1 ./g) * C1 * diagm(0=>1 ./g)
    @test run(sess, u.loc)≈μ1 .* (1 ./g)

    @test length(u1)==10
    @test size(u1)==(10,)
    @test size(u1,1)==10

    @test run(sess, u1[1].loc) ≈ [μ1[1]]
    @test run(sess, u1[1].cov) ≈ C1[1:1,1:1]

    @test run(sess, u1[1:3].loc) ≈ μ1[1:3]
    @test run(sess, u1[1:3].cov) ≈ C1[1:3,1:3]

    @test run(sess, u1[[2;4;3]].loc) ≈ μ1[[2;4;3]]
    @test run(sess, u1[[2;4;3]].cov) ≈ C1[[2;4;3],[2;4;3]]

    u = [u1;u1;u2]
    @test run(sess, u.loc)≈[μ1;μ1;μ2]
    B = zeros(30,30)
    B[1:10,1:10] = C1; B[11:20,11:20] = C1; B[21:30,21:30] = C2
    @test run(sess, u.cov)≈B
end