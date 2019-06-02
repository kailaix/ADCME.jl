@testset "xavier_init" begin
    @test_nowarn a = xavier_init([100,10], Float32)
end
