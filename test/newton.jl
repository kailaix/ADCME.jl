@testset "newton raphson with linesearch" begin
    function f1(θ, u)
        return u^2 - 1, 2*spdiag(u)
    end

    function f2(θ, u)
        return u^3-2u-5, spdiag(3u^2-2)
    end

    function f3(θ, u)
        return exp(2u)-u-6, spdiag(2*exp(2u)-1)
    end

    function f4(θ, u)
        return (cos(u)-u)^2, spdiag((-u + cos(u))*(-2*sin(u) - 2))
    end

    function f5(θ, u)
        return sum(1/3*(u-1)^3), (u - 1)^2, 2*spdiag(u-1)
    end

    function f6(θ, u)
        return sum((u-1.3)^2), 2(u-1.3), 2spdiag(length(u))
    end

    function f7(θ, u)
        return sum((0.988 * u^5 - 4.96 * u^4 + 4.978 * u^3 + 5.015 * u^2 - 6.043 * u - 1)),
                    4.94 * u^4 - 19.84 * u^3 + 14.934 * u^2 + 10.03 * u - 6.043,
                    spdiag(4.94*3*u^3 - 19.84*3*u^2 + 14.934*2*u + 10.03)
    end

    function f8(θ, u)
        return sum((u-1.3)^2), 2(u-1.3), 2spdiag(length(u))
    end

    res = [1.0, 2.0946, 0.97085, 0.73989]
    nr = Array{Any}(undef, 8)
    fs = [f1,f2,f3,f4,f5,f6,f7,f8]

    ADCME.options.newton_raphson.linesearch = true
    ADCME.options.newton_raphson.verbose = false
    ADCME.options.newton_raphson.rtol = 1e-12
    ADCME.options.newton_raphson.linesearch_options.αinitial = 1.0
    for i = 5:7
    nr[i] = newton_raphson(fs[i], constant(rand(1)), missing)
    end

    reset_default_options()
    ADCME.options.newton_raphson.verbose = false
    for i = 1:4
        nr[i] = newton_raphson(fs[i], constant(rand(1)), missing)
    end
    sess = Session(); init(sess)
    nrr = [run(sess, nr[i]) for i = 1:7]

    for i = 1:4
        @show nrr[i].x, nrr[i].iter, nrr[i].converged
        @test nrr[i].converged
    end

    for i = 5:7
        @show nrr[i].x, nrr[i].iter, nrr[i].converged
        @test nrr[i].converged
    end
end