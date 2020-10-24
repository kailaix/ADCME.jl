reset_default_graph(); sess = Session()

@testset "xavier_init" begin
    @test_nowarn a = xavier_init([100,10], Float32)
end

@testset "load_op" begin
    @test_nowarn load_op(ADCME.LIBADCME, "sparse_solver")
    @test_nowarn load_op_and_grad(ADCME.LIBADCME, "sparse_solver")
end


@testset "ae" begin
    n = ae_num([2,20,20,20,3])
    @test n==length(ae_init([2,20,20,20,3]))
end

@testset "register" begin 
    forward = x->log(1+exp(x))
    backward = (dy, y, x)->dy*(1-1/(1+y))
    f = register(forward, backward)
    x_ = rand(10)
    x = constant(x_)
    @test run(sess, f(x)) ≈ @. log(1+exp(x_))
    @test_nowarn run(sess,gradients(sum(f(x)), x))


    ADCME.options.newton_raphson.verbose = true
    function forward2(x, θ)
        f = (θ, y)->(y^3+1.0-θ -x, spdiag(3y^2))
        nr = newton_raphson(f, constant(ones(length(x))), θ)
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

@testset "list_physical_devices" begin 
    devices = list_physical_devices()   
    @test length(devices)>0
end

@testset "timestamp" begin 
    @test begin     
        a = constant(1.0)
        t0 = timestamp(a)
        sleep_time = sleep_for(a)
        t1 = timestamp(sleep_time)
        sess = Session(); init(sess)
        t0_, t1_ = run(sess, [t0, t1])
        time = t1_ - t0_
        true
    end
end

@testset "get_library_symbols" begin 
    @test length(get_library_symbols(ADCME.libadcme))>0
end