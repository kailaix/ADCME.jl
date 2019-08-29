@testset "sparse_constructor" begin
    A = sprand(10,10,0.1)
    s = SparseTensor(A)
    @test run(sess, s)≈A
    I = [1;2;4;2;3;5]
    J = [1;3;2;2;2;1]
    V = rand(6)
    A = sparse(I,J,V,6,5)
    s = SparseTensor(I,J,V,6,5)
    @test run(sess, s)≈A
    indices = [I J]
    s = SparseTensor(I,J,V,6,5)
    @test run(sess, s)≈A

    S = Array(s)
    @test run(sess, S)≈Array(A)

    @test size(s)==(6,5)
    @test size(s,1)==6
    @test size(s,2)==5
end

@testset "sparse_arithmetic" begin
    A1 = rand(10,5); B1 = sprand(10,5,0.3)
    A = constant(A1)
    B = SparseTensor(B1)
    @test run(sess, -B)≈-B1
    for op in [+,-]
        C = op(A, B)
        C1 = op(A1, B1)
        @test run(sess, C)≈C1
    end
    @test run(sess, B-A)≈B1-A1
end

@testset "sparse_adjoint" begin
    A = sprand(10,5,0.3)
    A1 = SparseTensor(A)
    @test run(sess, A1')≈sparse(A')
end

@testset "sparse_mul" begin
    A1 = rand(10,10); B1 = sprand(10,10,0.3)
    C1 = rand(10)
    A = constant(A1)
    B = SparseTensor(B1)
    C = constant(C1)
    @test run(sess, B*A) ≈ B1*A1
    @test run(sess, A*B) ≈ A1*B1
    @test run(sess, B*A1) ≈ B1*A1
    @test run(sess, A1*B) ≈ A1*B1
    @test run(sess, B*C) ≈ B1*C1
    @test run(sess, B*C1) ≈ B1*C1
end

@testset "sparse_vcat_hcat" begin
    B1 = sprand(10,3,0.3)
    B = SparseTensor(B1)
    @test run(sess, [B;B])≈[B1;B1]
    @test run(sess, [B B])≈[B1 B1]
end

@testset "sparse_indexing" begin
    B1 = sprand(10,10,0.3)
    B = SparseTensor(B1)
    @test run(sess, B[2:3,2:3])≈B1[2:3,2:3]
    @test run(sess, B[2:3,:])≈B1[2:3,:]
    @test run(sess, B[:,2:3])≈B1[:,2:3]

end

@testset "sparse_solve" begin
    A = sparse(I, 10,10) + sprand(10,10,0.1)
    b = rand(10)
    A1 = SparseTensor(A)
    b1 = constant(b)
    u = A1\b1
    @test run(sess, u) ≈ A\b
end