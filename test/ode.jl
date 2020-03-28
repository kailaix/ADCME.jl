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

@testset "alpha scheme" begin
    d0 = zeros(2)
    v0 = [2.;3.]
    a0 = zeros(2)
    Δt = 1e-3
    n = 9001
    Δt = Δt * ones(n)
    M = sparse([1.0 0.0;0.0 1.0])
    C = sparse(zeros(2,2))
    K = sparse([4.0 0.0;0.0 9.0])
    F = zeros(n-1, 2)

    d, v, a = αintegration(M, C, K, F, d0, v0, a0, Δt)

    d_, v_, a_ = run(sess, [d, v, a])

    tspan = 0:1e-3:(n-1)*1e-3
    @test norm(d_[:,1] - sin.(2tspan))<1e-3
    @test norm(d_[:,2] - sin.(3tspan))<1e-3
end 