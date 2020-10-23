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
    init(sess)

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
    v = get_variable(rand(10,10), name="test_get_variable")
    u = get_variable(rand(10,10), name="test_get_variable")
    @test u==v
    w = get_variable(Float64, shape=[10,20])
    @test size(w)==(10,20)
end


@testset "tensor" begin
    v = Variable(0.5)
    g = [1;v;0]
    G = [1 v 0; v 1 0; 0 0 (1-v)/2]
    gv = tensor(g;sparse=true)
    g_gv = gradients(2sum(gv), v)
    Gv = tensor(G;sparse=true)
    G_gv = gradients(sum(Gv), v)
    init(sess)
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
    init(sess)
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

@testset "cell" begin
    r = cell([[1.],[2.,3.]])
    @test run(sess, r[1])≈[1.0]
    @test run(sess, r[2])≈[2.,3.]
end

@testset "special matrices" begin 
    A = rand(10,10)
    @test run(sess, sym(A)) ≈ 0.5 * (A + A')
    @test all(eigvals(run(sess, spd(A))) .> 0.0)
end

@testset "ones/zeros like" begin 
    a = rand(100,10)
    b = ones_like(a)
    @test run(sess, b)≈ones(100,10)
    b = zeros_like(a)
    @test run(sess, b)≈zeros(100,10)
end

@testset "gradient_magnitude" begin 
    x = constant(rand(10))
    y = constant(rand(20))
    l = sum(x) + sum(y)

    gx, gy = gradients(l, [x,y])
    n = gradient_magnitude(l, [x, y])
    @test run(sess, n)≈sqrt(norm(run(sess, gx))^2 + norm(run(sess,gy))^2)
end

@testset "indexing with tensor" begin
    a = rand(10,3)
    i = constant(2)
    A = constant(a)
    @test run(sess, A[i,:])≈a[2,:]
    @test run(sess, A[i,3])≈a[2,3]
    @test run(sess, A[i,[1;2]])≈a[2,[1;2]]
    @test run(sess, A[:, i])≈a[:, 2]
    @test run(sess, A[3, i])≈a[3, 2]
    @test run(sess, A[[1;2], i])≈a[[1;2], 2]
end

@testset "ndims" begin 
    @test ndims(constant(0.0))==0
    @test ndims(constant(rand(2)))==1
    @test ndims(constant(rand(3,3)))==2
    @test ndims(constant(rand(4,4,4)))==3
end

@testset "gradients_colocate" begin
    @cpu 1 begin
        global a = constant(rand(10,10))
        global b = 2a 
    end
    loss = sum(b)
    g = gradients_colocate(loss, a)
    @test g.device == "/device:CPU:1"
end



@testset "is_variable" begin 
    a = Variable(rand(10))
    b = constant(rand(10))
    @test is_variable(a)==true 
    @test is_variable(b)==false
end