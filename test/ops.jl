@testset "*" begin
    A = Array{Any}(undef, 6)
    A[1] = rand(Float32,10,10)
    A[2] = rand(Float32,10)
    A[3] = rand(Float32)
    A[4] = Variable(A[1])
    A[5] = Variable(A[2])
    A[6] = Variable(A[3])
    run(sess, global_variables_initializer())
    for i = 1:6
        for j = 1:6
            if i<=3 && j<=3 || (i in [2,5] && j in [1,4]) ||
                (i in [2,5] && j in [2,5])
                continue
            end
            ret1 = A[i]*A[j]
            ret1 = run(sess, ret1)
            ret2 = A[i>3 ? i-3 : i] * A[j>3 ? j-3 : j] 
            @test ret1 ≈ ret2
        end
    end

    @test A[1].*A[1] ≈ run(sess, A[4].*A[4])
    @test A[1].*A[1] ≈ run(sess, A[4].*A[1])
    @test A[1].*A[1] ≈ run(sess, A[1].*A[4])
    @test A[2].*A[2] ≈ run(sess, A[2].*A[5])
    @test A[2].*A[2] ≈ run(sess, A[5].*A[5])
    @test A[2].*A[2] ≈ run(sess, A[5].*A[2])
end

@testset "reshape" begin
    a = constant([1 2 3;4 5 6.])
    @test run(sess, reshape(a, 6))≈[ 1.0
    4.0
    2.0
    5.0
    3.0
    6.0]

    @test run(sess, reshape(a, 3, 2))≈[1.0  5.0
                                        4.0  3.0
                                        2.0  6.0]
    
    b = constant([1;2;3;4;5;6])
    @test run(sess, reshape(b, 2, 3))≈[1  3  5
                                    2  4  6]
    @test run(sess, reshape(constant(2.),1 )) ≈[2.0]

    a = constant([0.136944  0.364238
    0.250264  0.202175
    0.712224  0.388285
    0.163095  0.797934
    0.740991  0.382738])
    @test run(sess, reshape(a, :, 1))≈reshape([0.136944  0.364238
    0.250264  0.202175
    0.712224  0.388285
    0.163095  0.797934
    0.740991  0.382738], :, 1)
    @test run(sess, reshape(a, 1,:))≈reshape([0.136944  0.364238
    0.250264  0.202175
    0.712224  0.388285
    0.163095  0.797934
    0.740991  0.382738], 1,:)
    
end


@testset "scatter_update" begin
    A = Variable(ones(3))
    A = scatter_add(A, 3, 2.)
    B = [1.;1.;3.]

    C = Variable(ones(9))
    D = scatter_add(C, [2], [1], 1., 3, 3)
    G = reshape(D, 3, 3)
    E = [1. 1. 1; 2. 1. 1.; 1. 1. 1.]
    run(sess, global_variables_initializer())
    @test run(sess, A)≈B
    @test run(sess, G)≈E

    A = Variable(rand(10))
    B = Variable(rand(5))
    C = scatter_add(A, [1;4;5;6;7], 2B)
    D = gradients(sum(C), B)
    E = gradients(sum(C), A)
    run(sess, global_variables_initializer())
    @test run(sess, D)≈2ones(5)
    @test run(sess, E)≈ones(10)

    A = Variable(ones(10,10))
    B = Variable(ones(3,4))
    C = scatter_add(A, [1;2;3],[3;4;5;6], B)
    run(sess, global_variables_initializer())
    @test run(sess, C)≈[1.0  1.0  2.0  2.0  2.0  2.0  1.0  1.0  1.0  1.0
    1.0  1.0  2.0  2.0  2.0  2.0  1.0  1.0  1.0  1.0
    1.0  1.0  2.0  2.0  2.0  2.0  1.0  1.0  1.0  1.0
    1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
    1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
    1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
    1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
    1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
    1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
    1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0]


    a = rand(10)
    A = constant(a)
    A = scatter_add(A, 3:4, -A[3:4])
    a[3:4] .= 0.0
    @test run(sess, A) ≈ a
end

@testset "scatter_update_pyobject" begin
    A = constant(ones(10))
    B = constant(ones(3))
    ind = constant([2;4;6], dtype=Int32)
    C = scatter_add(A, ind, B)
    @test run(sess, C)≈[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0]
    @test run(sess, gradients(sum(C),B))≈ones(3)
    @test run(sess, gradients(sum(C),A))≈ones(10)

    C = scatter_sub(A, ind, B)
    @test run(sess, C)≈[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    @test run(sess, gradients(sum(C),B))≈-ones(3)
    @test run(sess, gradients(sum(C),A))≈ones(10)

    B = constant(3*ones(3))
    C = scatter_update(A, ind, B)
    @test run(sess, C)≈[1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0]
    @test run(sess, gradients(sum(C),B))≈ones(3)
    @test run(sess, gradients(sum(C),A))≈[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
end

@testset "Operators" begin
    A = rand(10,2)
    TA = Variable(A)
    a = Variable(1.0)
    b = Variable(0.5)

    maxA = argmax(A, dims=2)
    maxA = [maxA[i][2] for i = 1:10]
    maxTA = argmax(TA, dims=2)

    minA = argmin(A, dims=2)
    minA = [minA[i][2] for i = 1:10]
    minTA = argmin(TA, dims=2)

    B = rand(Float32, 10)
    TB = Variable(B)
    TBc = constant(B)

    gop = group_assign([a,b], [2.0,2.0])
    run(sess, global_variables_initializer())
    @test maxA  ≈ run(sess, maxTA)
    @test minA  ≈ run(sess, minTA)
    @test get_dtype(cast(maxTA, Float64))==Float64
    @test B.^2 ≈ run(sess, TB^2)
    @test B.^2 ≈ run(sess, TBc^2)

    @test 1.0 ≈ run(sess, max(a, b))
    @test 0.5 ≈ run(sess, min(a, b))
    @test maximum(A) ≈ run(sess, maximum(TA))
    @test minimum(A) ≈ run(sess, minimum(TA))
    @test reshape(maximum(A, dims=1), 2) ≈ run(sess, maximum(TA, dims=1))
    @test reshape(minimum(A, dims=1),2) ≈ run(sess, minimum(TA, dims=1))
    @test reshape(maximum(A, dims=2),10) ≈ run(sess, maximum(TA, dims=2))
    @test reshape(minimum(A, dims=2),10) ≈ run(sess, minimum(TA, dims=2))

    run(sess, gop)
    @test run(sess, a)≈2.0 && run(sess,b) ≈ 2.0
end

@testset "Other Operators" begin
    A = Variable(rand(10,10))
    @test_nowarn group(assign(A, rand(10,10)), A+A, 2*A)
end

@testset "Concat and stack" begin
    A = Variable(rand(10))
    B = Variable(rand(10,3))
    @test size(stack([A, A]))==(2,10)
    @test size(concat([A, A], dims=0))==(20,)
    @test size([A;A])==(20,)
    @test size([A A])==(10,2)
    @test size([B;B])==(20,3)
    @test size([B B])==(10,6)
end


@testset "Vectorize" begin
    A = rand(10)
    B = reshape(A, 2, 5)
    tA = Variable(A)
    tB = Variable(B)
    rA = rvec(tA)
    cB = vec(tB)
    init(sess)
    a, b = run(sess, [rA, cB])
    @test a≈reshape(A, 1, 10)
    @test b≈A
end

@testset "Solve" begin
    A = rand(10,10)
    x = rand(10)
    y = A*x
    tA = constant(A)
    ty = constant(y)
    @test run(sess, tA\ty)≈x
    @test run(sess, A\ty)≈x
    @test run(sess, tA\y)≈x
end

@testset "diff" begin
    A = rand(10,10)
    a = rand(10)
    tfA = constant(A)
    tfa = constant(a)
    @test run(sess, diff(tfA, dims=1))≈diff(A, dims=1)
    @test run(sess, diff(tfA, dims=2))≈diff(A, dims=2)
    @test run(sess, diff(tfa))≈diff(a)
end

@testset "clip" begin
    a = constant(3.0)
    a = clip(a, 1.0, 2.0)
    @test run(sess,a)≈2.0
end

@testset "map" begin
    a = constant([1.0,2.0,3.0])
    b = map(x->x^2, a)
    @test run(sess, b)≈[1.0;4.0;9.0]
end

@testset "diag" begin
    C = rand(3,3)
    D = rand(3)
    @test run(sess, diag(constant(C))) ≈  diag(C)
    @test run(sess, diagm(constant(D))) ≈ diagm(0=>D)
end

@testset "dot" begin
    A = rand(10)
    B = rand(10)
    tA = constant(A); tB = constant(B)
    u = sum(A.*B)
    @test run(sess, dot(tA, tB))≈u
    @test run(sess, dot(A, tB))≈u
    @test run(sess, dot(tA, B))≈u
end

@testset "prod" begin
    A = rand(10)
    @test run(sess, prod(constant(A)))≈prod(A)
end

@testset "findall" begin
    a = constant([0.849268    0.888376  0.0928501  0.119263  0.614649
    0.768562    0.978984  0.0802774  0.764186  0.491608
    0.00877878  0.751463  0.99539    0.613257  0.070678])
    b = a>0.5
    bi = findall(b)
    @test run(sess, bi)==[1  1
    1  2
    1  5
    2  1
    2  2
    2  4
    3  2
    3  3
    3  4]
    ci = findall(b[1,:])
    @test run(sess, ci)==[1;2;5]
end

@testset "svd" begin
    A = rand(10,20)
    r = svd(constant(A))
    A2 = r.U*diagm(r.S)*r.Vt 
    @test run(sess, A2)≈A
end