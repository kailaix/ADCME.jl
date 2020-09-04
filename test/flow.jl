reset_default_graph(); sess = Session()
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
    flows = [SlowMAF(dim, false, [x->fc(x, [20,20,2], "fc1"); x->fc(x,[20,20,2],"fc2")])]
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
    @test_skip begin
        dim = 6
        K = 8
        n = dim÷2
        flows = [NeuralCouplingFlow(dim, x->fc(x, [20,20,n*(3K-1)], "ncf1"), x->fc(x,[20,20,n*(3K-1)],"ncf2"), K)]
        x1, x2, j1, j2 = test_flow(sess, flows, dim)
        @test x1≈x2
        @test j1≈-j2
    end
end

@testset "Permute" begin 
    dim = 6
    flows = [Permute(dim, randperm(6) .- 1)]
    x1, x2, j1, j2 = test_flow(sess, flows, dim)
    @test x1≈x2
    @test j1≈-j2
end

@testset "composite" begin 
    function create_linear_transform(dim)
        permutation = randperm(dim) .- 1
        flow = [Permute(dim, permutation);LinearFlow(dim)]
    end
    dim = 5
    flows = create_linear_transform(dim)
    x1, x2, j1, j2 = test_flow(sess, flows, dim)
    @test x1≈x2
    @test j1≈-j2
end

# @testset "composite 2" begin 
#     function create_base_transform(dim, K=8)
#         n1 = dim÷2
#         n2 = dim - n1
#         r1 = x->Resnet1D(n1 * (3K-1), 256, dropout_probability=0.0, use_batch_norm=false)(x)
#         r2 = x->Resnet1D(n2 * (3K-1), 256, dropout_probability=0.0, use_batch_norm=false)(x)
#         NeuralCouplingFlow(dim, r1, r2)
#     end
#     dim = 6
#     flows = [create_base_transform(dim)]
#     x1, x2, j1, j2 = test_flow(sess, flows, dim)
#     @test x1≈x2
#     @test j1≈-j2
# end