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