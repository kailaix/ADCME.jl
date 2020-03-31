@testset "xavier_init" begin
    @test_nowarn a = xavier_init([100,10], Float32)
end

@testset "load_op" begin
    @test begin
        # ADCME.install_custom_op_dependency()
        ADCME.compile("SparseSolver")
        # somehow we cannot first call `load_op_and_grad` and then call `load_op` 
        load_op("$(@__DIR__)/../deps/CustomOps/SparseSolver/build/libSparseSolver", "sparse_solver")
        load_op_and_grad("$(@__DIR__)/../deps/CustomOps/SparseSolver/build/libSparseSolver", "sparse_solver")
        true
    end
end


@testset "install" begin
    install("OTNetwork", force=true)
end

@testset "ae" begin
    n = ae_num([2,20,20,20,3])
    @test n==length(ae_init([2,20,20,20,3]))
end

@test "register" begin 
    forward = x->log(1+exp(x))
    backward = (dy, y, x)->dy*(1-1/(1+y))
    f = register(forward, backward)
    x_ = rand(10)
    x = constant(x_)
    @test run(sess, f(x)) ≈ @. log(1+exp(x_))
    @test_nowarn run(sess,gradients(sum(f(x)), x))


    function forward2(x, θ)
        f = (θ, y)->(y^3+1.0-θ -x, spdiag(3y^2))
        nr = newton_raphson(f, constant(ones(length(x))), θ, options=Dict("verbose"=>true))
        y = nr.x 
        return y 
    end

    function backward2(dy, y, x, θ)
        e = 1/3*(θ+x-1)^(-2/3)
        e, e
    end

    f2 = register(forward2, backward2)

    θ = constant(ones(5))
    x = constant(8ones(5))
    y = f2(x, θ)
    @test run(sess, y)≈ones(5)*2

    @test run(sess, gradients(sum(y), θ))≈0.08333333333333333*ones(5)
end