@testset "xavier_init" begin
    @test_nowarn a = xavier_init([100,10], Float32)
end

@testset "load_op" begin
    @test_skip begin
        ADCME.install_custom_op_dependency()
        ADCME.compile("SparseSolver")
        # somehow we cannot first call `load_op_and_grad` and then call `load_op` 
        load_op("$(@__DIR__)/../deps/CustomOps/SparseSolver/build/libSparseSolver", "sparse_solver")
        load_op_and_grad("$(@__DIR__)/../deps/CustomOps/SparseSolver/build/libSparseSolver", "sparse_solver")
    end
end

@testset "timer" begin
    @test_skip begin
        A = constant(rand(1000,1000))
        A = tic(A)
        r = svd(A)
        a, t = toc(r.U)
        run(sess, a)
        @test run(sess, t)>0.0
    end
end