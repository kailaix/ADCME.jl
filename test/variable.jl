using LinearAlgebra
@testset "indexing" begin
    A1 = rand(10)
    A2 = rand(10,10)
    B1 = Variable(A1)
    B2 = Variable(A2)

    ind = Array{Any}(undef, 4)
    ind[1] = 3
    ind[2] = 2:4
    ind[3] = zeros(Bool,10)
    ind[3][3:4] .= true
    ind[4] = [3;4;5]
    run(sess, global_variables_initializer())

    for i = 1:4
        @test A1[ind[i]] ≈ run(sess, B1[ind[i]])
        @test A2[ind[i], :] ≈ run(sess, B2[ind[i], :])
        @test A2[:, ind[i]] ≈ run(sess, B2[:, ind[i]])
    end

    for i = 1:4
        for j = 1:4
            @test A2[ind[i], ind[j]] ≈ run(sess, B2[ind[i], ind[j]])
        end
    end
end

@testset "Variables" begin
    # @test_nowarn placeholder(Float64, shape=[nothing,20], name="myvar")
    # @test_nowarn Variable(rand(10,10), dtype=Float64, name="myvar")
    # @test_nowarn constant(rand(10,10), dtype=Float64, name="myvar")
    # @test_nowarn get_variable("W", shape=[10,20], dtype=Float64)
    # @test_nowarn placeholder(Float64, shape=[nothing,20])
    # @test_nowarn Variable(rand(10,10), dtype=Float64)
    # @test_nowarn constant(rand(10,10), dtype=Float64)
    @test get_dtype(constant(rand(10,10))) == Float64
    @test get_dtype(Variable(rand(Int64,10,10))) == Int64
end


@testset "tensor" begin
    v = Variable(0.5)
    g = [1;v;0]
    G = [1 v 0; v 1 0; 0 0 (1-v)/2]
    gv = tensor(g;sparse=true)
    g_gv = gradients(2sum(gv), v)
    Gv = tensor(G;sparse=true)
    G_gv = gradients(sum(Gv), v)
    run(sess, global_variables_initializer())
    @test run(sess, g_gv) ≈ 2.0
    @test run(sess, G_gv) ≈ 1.5
end

@testset "Hessian" begin
    x = Variable(randn(), dtype = Float32)
    y = Variable(randn(), dtype = Float32)
    g = stack([x,y])
    f = g[1]^2 + 2*g[1]*g[2] + 3*g[2]^2 + 4*g[1] + 5*g[2] + 6
    hess = hessian(f, g)
    gv = rand(Float32,2)
    hess_v =  hessian_vector(f, g, gv)
    val = [[2.;2.] [2.;6.]] * gv
    run(sess, global_variables_initializer())
    @test run(sess, hess) ≈ [[2.;2.] [2.;6.]]
    @test run(sess, hess_v)≈val
end

@testset "Jacobian" begin
    xs = constant(rand(10))
    A = rand(20,10)
    ys = A*xs
    jac = gradients(ys, xs)
    @test run(sess, jac)≈A
end

@testset "gradients_v" begin
    xs = constant(1.0)
    ys = stack([xs,xs,xs,xs])
    gv = gradients(ys, xs)
    @test run(sess, gv)≈[1.0;1.0;1.0;1.0]
end

@testset "size and length" begin
    a = Variable(1.0)
    b = Variable(rand(10))
    c = Variable(rand(10,20))
    @test size(a)==()
    @test size(b)==(10,)
    @test size(b,1)==10
    @test size(c) == (10,20)
    @test size(c,1)==10
    @test size(c,2)==20
    @test length(a)==1
    @test length(b)==10
    @test length(c)==200
end

@testset "copy" begin
    o = constant(rand(10,10))
    v = copy(o)
    init(sess)
    @test run(sess, v)≈run(sess, o)
end

@testset "getindex" begin
     i = constant(2)
     a = constant([1.;2.;3.;4.])
     @test run(sess, a[i])≈2.0
end


@testset "convert_to_tensor" begin
    a = rand(10)
    @test run(sess, convert_to_tensor(a))≈a
    a = 5.0
    @test run(sess, convert_to_tensor(a))≈a
    a = constant(rand(10))
    @test run(sess, convert_to_tensor(a))≈run(sess, a)
    @test ismissing(convert_to_tensor(missing))
    @test isnothing(convert_to_tensor(nothing))
end