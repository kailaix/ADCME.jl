reset_default_graph(); sess = Session()
@testset "customop" begin
    if isdir("temp")
        rm("temp", force=true, recursive=true)
    end
    mkdir("temp")
    cd("temp")    
    customop()
    customop()
    customop(;julia=true)
    cp("$(@__DIR__)/../examples/while_loop/DirichletBD/DirichletBD.cpp", "DirichletBD.cpp", force=true)
    cp("$(@__DIR__)/../examples/while_loop/DirichletBD/DirichletBD.h", "DirichletBD.h", force=true)
    cp("$(@__DIR__)/../examples/while_loop/DirichletBD/CMakeLists.txt", "CMakeLists.txt", force=true)
    mkdir("build")
    cd("build")
    ADCME.cmake()
    ADCME.make()
    cd("..")
    rm("temp", force=true, recursive=true)
    @test true
end