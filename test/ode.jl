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
    F = zeros(n, 2)

    d, v, a = αscheme(M, C, K, F, d0, v0, a0, Δt)
    d_, v_, a_ = run(sess, [d, v, a])

    tspan = 0:1e-3:(n-1)*1e-3
    @test norm(d_[1:end-1,1] - sin.(2tspan))<1e-3
    @test norm(d_[1:end-1,2] - sin.(3tspan))<1e-3
end 

@testset "tr_bdf2" begin 
    D0 = sparse([1.0 0.0;1.0 1.0])
    D1 = sparse([2.0 0.0; 0.0 1.0])
    ts = collect(LinRange(0, 1, 100))
    Δt = ts[2]-ts[1]
    F = zeros(199, 2)
    for i = 1:100
        t = ts[i]
        F[2*i-1, :] = [
           t^2 + 5t + 2 
           t^2 + 2t + 1
        ]
        if i<100
            t = t + Δt/2
            F[2*i, :] = [
                t^2 + 5t + 2 
                t^2 + 2t + 1
            ]
        end
    end
    td = TR_BDF2(D0, D1, Δt)
    @test td.symbolic == false 
    u = td(zeros(2), F)
    
    @test abs(u[end, 1]-2.0)<1e-5
    @test abs(u[end, 2]-1.0)<1e-5

    td = constant(td)
    ua = td(zeros(2), F)
    @test run(sess, ua)≈u
end

@testset "ExplicitNewmark" begin 
    
end 