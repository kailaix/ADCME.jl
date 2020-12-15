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

@testset "sparse_assembler" begin
        
    m = 20
    n = 100
    handle = SparseAssembler(100, m, 0.0)
    op = PyObject[]
    A = zeros(m, n)
    for i = 1:1
        ncol = rand(1:n, 10)
        row = rand(1:m)
        v = rand(10)
        for (k,val) in enumerate(v)
            @show k
            A[row, ncol[k]] += val
        end
        @show v
        push!(op, accumulate(handle, row, ncol, v))
    end
    op = vcat(op...)
    J = assemble(m, n, op)
    B = run(sess, J)
    @test norm(A-B)<1e-8



    handle = SparseAssembler(100, 5, 1.0)
    op1 = accumulate(handle, 1, [1;2;3], [2.0;0.5;0.5])
    J = assemble(5, 5, op1)
    B = run(sess, J)
    @test norm(B-[2.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0])<1e-8

    handle = SparseAssembler(100, 5, 0.0)
    op1 = accumulate(handle, 1, [1;1], [1.0;1.0])
    op2 = accumulate(handle, 1, [1;2], [1.0;1.0])

    J = assemble(5, 5, [op1;op2])
    B = run(sess, J)
    @test norm(B-[3.0  1.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0])<1e-8

        
end


@testset "sparse_least_square" begin
    ii = Int32[1;1;2;2;3;3]
    jj = Int32[1;2;1;2;1;2]
    vv = Float64[1;2;3;4;5;6]
    ff = Float64[1;1;1]
    A = SparseTensor(ii, jj, vv, 3, 2)
    o = A\ff
    @test norm(run(sess, o)-[-1;1])<1e-6
end

@testset "sparse mat mul" begin
    A = sprand(10,5,0.3)
    B = sprand(5,20,0.3)
    C = A*B
    CC = SparseTensor(A)*SparseTensor(B)
    C_ = run(sess, CC)
    @test C_≈C

    A = spdiagm(0=>[1.;2.;3;4;5])
    B = sprand(5,20,0.3)
    C = A*B
    CC = SparseTensor(A)*SparseTensor(B)
    C_ = run(sess, CC)
    @test C_≈C

    A = sprand(10,5,0.5)
    B = spdiagm(0=>[1.;2.;3;4;5])
    C = A*B
    CC = SparseTensor(A)*SparseTensor(B)
    C_ = run(sess, CC)
    @test C_≈C
end

@testset "spdiag" begin
    p = rand(10)
    A = spdiagm(0=>p)
    B = spdiag(constant(p))
    C = spdiag(10)
    @test run(sess, B)≈A
    @test B._diag
    @test run(sess, C)≈spdiagm(0=>ones(10))
end

@testset "spzero" begin
    q = spzero(10)
    @test run(sess, q)≈sparse(zeros(10,10))
    q = spzero(10,20)
    @test run(sess, q)≈sparse(zeros(10,20))
end

@testset "sparse indexing" begin
    i1 = unique(rand(1:20,3))
    j1 = unique(rand(1:30,3))
    A = sprand(20,30,0.3)
    Ad = Array(A[i1, j1])
    B = SparseTensor(A)
    Bd = Array(B[i1, j1])
    Bd_ = run(sess, Bd)
    @test Ad≈Bd_
end

@testset "sum" begin
    s = sprand(10,20,0.2)
    S = SparseTensor(s)
    @test run(sess, sum(S)) ≈ sum(s)
    @test run(sess, sum(S,dims=1)) ≈ sum(Array(s),dims=1)[:]
    @test run(sess, sum(S,dims=2)) ≈ sum(Array(s),dims=2)[:]
end

@testset "dense_to_sparse" begin
    A = sprand(10,20,0.3)
    B = Array(A)
    @test run(sess, dense_to_sparse((B))) ≈ A
    @test run(sess, dense_to_sparse(constant(B))) ≈ A
end

@testset "spdiagm" begin
    a = rand(10)
    b = rand(9)
    A = spdiag(
        10, 
        0=>a, 
        -1=>b
    )
    B = diagm(
        0=>a, 
        -1=>b
    )
    @test Array(run(sess, A)) ≈ B

    b = rand(7)
    A = spdiag(
        10,  
        0=>a, 
        -3=>b
    )
    B = diagm(
        0=>a, 
        -3=>b
    )
    @test Array(run(sess, A)) ≈ B

    b = rand(7)
    A = spdiag(
        10,  
        0=>a, 
        -3=>b,
        3=>4b
    )
    B = diagm(
        0=>a, 
        -3=>b,
        3=>4b
    )
    @test Array(run(sess, A)) ≈ B

    b = rand(7)
    A = spdiag(
        10,  
        0=>a, 
        -3=>b
    )
    B = diagm(
        0=>a, 
        -3=>b
    )
    @test Array(run(sess, A)) ≈ B
end

@testset "hvcat" begin
    A = sprand(10,5,0.3)
    B = sprand(10,5,0.2)
    C = sprand(5,10,0.4)
    D = [A B;C]
    D_ = [SparseTensor(A) SparseTensor(B); SparseTensor(C)]
    @test run(sess, D_)≈D
end

@testset "find" begin 
    A = sprand(10,10, 0.3)
    ii = Int64[]
    jj = Int64[]
    vv = Float64[]
    for i = 1:10
        for j = 1:10
            if A[i,j]!=0
                push!(ii, i)
                push!(jj, j)
                push!(vv, A[i,j])
            end
        end
    end
    a = SparseTensor(A)
    i, j, v = find(a)
    @test run(sess, i)≈ii
    @test run(sess, j)≈jj
    @test run(sess, v)≈vv

    @test run(sess, rows(a))≈ii
    @test run(sess, cols(a))≈jj
    @test run(sess, values(a))≈vv
end

@testset "sparse scatter update add" begin
    A = sprand(10,10,0.3)
    B = sprand(3,3,0.6)
    ii = [1;4;5]
    jj = [2;4;6]
    u = scatter_update(A, ii, jj, B)
    C = copy(A)
    C[ii,jj] = B
    @test run(sess, u)≈C

    u = scatter_add(A, ii, jj, B)
    C = copy(A)
    C[ii,jj] += B
    @test run(sess, u)≈C
end

@testset "constant sparse" begin 
    A = sprand(10,10,0.3)
    B = constant(A)
    @test run(sess, B)≈A
end

@testset "get index" begin 
    idof = [false;true]
    M = spdiag(constant(ones(2)))
    Md = M[idof, idof]
    @test run(sess, Md) ≈ sparse(reshape([1.0],1,1))
end

@testset "sparse_factorization_and_solve" begin 
    A = sprand(10,10,0.7)
    rhs1 = rand(10)
    rhs2 = rand(10)
    Afac = factorize(constant(A))
    v1 = Afac\rhs1
    v2 = Afac\rhs2

    @test norm(run(sess, v1) - A\rhs1)<1e-8
    @test norm(run(sess, v2) - A\rhs2)<1e-8
end

@testset "sparse solver warning" begin 
    A = SparseTensor(zeros(10,10))
    b = A\rand(10)
    @test_throws PyCall.PyError run(sess, b)
end

@testset "sparse promote" begin 
    a = sprand(10,10,0.3)
    b = spdiag(constant(rand(10)))
    @test_nowarn a+b
    @test_nowarn a*b
    @test_nowarn a-b
    @test_nowarn b+a
    @test_nowarn b-a
    @test_nowarn b*a
end

@testset "trisolve" begin 
    n = 10
    a = rand(n-1)
    b = rand(n).+10
    c = rand(n-1)
    d = rand(n)

    A = diagm(0=>b, -1=>a, 1=>c)
    x = A\d

    u = trisolve(a,b,c,d)
    @test norm(run(sess, u)-x)<1e-3
end

@testset "compress" begin 
    indices = [
        1 1 
        1 1
        2 2
        3 3
    ]
    v = [1.0;1.0;1.0;1.0]
    A = SparseTensor(indices[:,1], indices[:,2], v, 3, 3)
    Ac = compress(A)
    sess = Session(); init(sess)


    @test run(sess, Ac.o.indices) == [0 0;1 1;2 2]
    @test run(sess, Ac.o.values) ≈ [2.0;1.0;1.0]
end