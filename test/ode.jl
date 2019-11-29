@testset "runge_kutta" begin
    function f(t, y, θ)
        return y
    end

    x0 = 1.0
    out = rk4(f, 1.0, 100, x0)
    @test abs(run(sess, out)[end]-exp(1))<1e-5
    out = ode45(f, 1.0, 100, x0)
    @test abs(run(sess, out)[end]-exp(1))<1e-5


    function f(t, y, θ)
        return y
    end

    x0 = [1.0;1.0]
    out = rk4(f, 1.0, 100, x0)
    @test norm(run(sess, out)[end,:] .- exp(1))<1e-5
    out = ode45(f, 1.0, 100, x0)
    @test norm(run(sess, out)[end,:] .- exp(1))<1e-5

    x0 = [1.0 1.0
          1.0 1.0]
    out = rk4(f, 1.0, 100, x0)
    @test norm(run(sess, out)[end,:,:] .- exp(1))<1e-5
    out = ode45(f, 1.0, 100, x0)
    @test norm(run(sess, out)[end,:,:] .- exp(1))<1e-5

    
end