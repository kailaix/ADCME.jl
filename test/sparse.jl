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
    @test_skip begin
        A = sparse(I, 10,10) + sprand(10,10,0.1)
        b = rand(10)
        A1 = SparseTensor(A)
        b1 = constant(b)
        u = A1\b1
        run(sess, u) ≈ A\b
    end 
end

@testset "sparse_assembler" begin
    @test_skip begin
        accumulator, creater, initializer = SparseAssembler()
        initializer(5)
        op1 = accumulator(1, [1;2;3], ones(3))
        op2 = accumulator(1, [3], [1.])
        op3 = accumulator(2, [1;3], ones(2))
        run(sess, [op1,op2,op3])
        ii,jj,vv = creater()
        i,j,v = run(sess, [ii,jj,vv])
        A = sparse(i,j,v,5,5)
        @test Array(A)≈[1.0  1.0  2.0  0.0  0.0
                        1.0  0.0  1.0  0.0  0.0
                        0.0  0.0  0.0  0.0  0.0
                        0.0  0.0  0.0  0.0  0.0
                        0.0  0.0  0.0  0.0  0.0]
    end
end

@testset "sparse_least_square" begin
    @test_skip begin
        ii = Int32[1;1;2;2;3;3]
        jj = Int32[1;2;1;2;1;2]
        vv = Float64[1;2;3;4;5;6]
        ff = Float64[1;1;1]
        A = SparseTensor(ii, jj, vv, 3, 2)
        o = A\ff
        @test norm(run(sess, o)-[-1;1])<1e-6
    end
end