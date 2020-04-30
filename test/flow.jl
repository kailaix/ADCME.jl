function test_flow(sess, flows, dim)
    prior = ADCME.MultivariateNormalDiag(loc=zeros(dim))
    model = NormalizingFlowModel(prior, flows)
    x = rand(3, dim)
    z, _, J = model(x)
    xs, iJ = ADCME.backward(model, z[end])
    init(sess)
    x_, J_, iJ_ = run(sess, [xs[end], J, iJ])
    x, x_, J_, iJ_
end


@testset "LinearFlow" begin 
    dim = 3
    flows = [LinearFlow(dim)]
    x1, x2, j1, j2 = test_flow(sess, flows, dim)
    @test x1≈x2
    @test j1≈-j2
end


@testset "AffineConstantFlow" begin 
    dim = 3
    flows = [AffineConstantFlow(dim)]
    x1, x2, j1, j2 = test_flow(sess, flows, dim)
    @test x1≈x2
    @test j1≈-j2
end

@testset "ActNorm" begin 
    dim = 3
    flows = [ActNorm(dim)]
    x1, x2, j1, j2 = test_flow(sess, flows, dim)
    @test x1≈x2
    @test j1≈-j2
end



@testset "SlowMAF" begin 
    dim = 3
    flows = [SlowMAF(dim, false, [x->fc(x, [20,20,2]); x->fc(x,[20,20,2],"fc2")])]
    x1, x2, j1, j2 = test_flow(sess, flows, dim)
    @test x1≈x2
    @test j1≈-j2
end


@testset "MAF" begin 
    dim = 3
    flows = [MAF(dim, false, [20,20,20])]
    x1, x2, j1, j2 = test_flow(sess, flows, dim)
    @test x1≈x2
    @test j1≈-j2
end

@testset "IAF" begin 
    dim = 3
    flows = [IAF(dim, false, [20,20,20])]
    x1, x2, j1, j2 = test_flow(sess, flows, dim)
    @test x1≈x2
    @test j1≈-j2
end


@testset "Invertible1x1Conv" begin 
    dim = 3
    flows = [Invertible1x1Conv(dim)]
    x1, x2, j1, j2 = test_flow(sess, flows, dim)
    @test x1≈x2
    @test j1≈-j2
end

@testset "AffineHalfFlow" begin 
    dim = 3
    flows = [AffineHalfFlow(dim, true)]
    x1, x2, j1, j2 = test_flow(sess, flows, dim)
    @test x1≈x2
    @test j1≈-j2
end


@testset "NeuralCouplingFlow" begin 
    dim = 6
    K = 8
    n = dim÷2
    flows = [NeuralCouplingFlow(dim, x->fc(x, [20,20,n*(3K-1)], "ncf1"), x->fc(x,[20,20,n*(3K-1)],"ncf2"), K)]
    x1, x2, j1, j2 = test_flow(sess, flows, dim)
    @test x1≈x2
    @test j1≈-j2
end