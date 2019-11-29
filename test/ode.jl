@testset "runge_kutta" begin
    function f(t, y, θ)
        return y
    end

    x0 = 1.0
    out = runge_kutta(x0, 1.0, 100, f)
    @test abs(run(sess, out)[end]-exp(1))<1e-5

    function f(t, y, θ)
        return y
    end

    x0 = [1.0;1.0]
    out = runge_kutta(x0, 1.0, 100, f)
    @test norm(run(sess, out)[end,:] .- exp(1))<1e-5

    x0 = [1.0 1.0
          1.0 1.0]
    out = runge_kutta(x0, 1.0, 100, f)
    @test norm(run(sess, out)[end,:,:] .- exp(1))<1e-5

    
end