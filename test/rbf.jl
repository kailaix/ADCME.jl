@testset "radial basis function" begin 
    xc = rand(10)
    yc = rand(10)
    e = rand(10)
    c = rand(10)
    d = rand(3)
    r = RBF2D(xc, yc; c=c, eps=e, d=d, kind = 0)

    x = rand(5)
    y = rand(5)
    o = r(x, y)

    O = zeros(5)
    for i = 1:5
        for j = 1:10
            dist = sqrt((x[i]-xc[j])^2 + (y[i]-yc[j])^2)
            O[i] += c[j] * exp(-(e[j]*dist)^2)
        end
        O[i] += d[1] + d[2] * x[i] + d[3] * y[i]
    end
    init(sess)
    @test norm(run(sess, o)-O)<1e-5
end

@testset "interp1" begin 
    Random.seed!(233)
    x = sort(rand(10))
    y = @. x^2 + 1.0
    z = [x[1]; x[2]; rand(5) * (x[end]-x[1]) .+ x[1]; x[end]]
    u = interp1(x,y,z)
    @test norm(run(sess, u)-[1.026422850882909
                    1.044414684090653
                    1.312604319732756
                    1.810845361128137
                    1.280789421523103
                    1.600084940795178
                    1.930560200260898
                    1.972130181835701]) < 1e-10
end


@testset "rbf3d" begin 

    function f0(r, e)
        return exp(-(e*r)^2)
    end

    function f1(r, e)
        return sqrt((1+(e*r)^2))
    end

    function f2(r, e)
        return 1/(1+(e*r)^2)
    end

    function f3(r, e)
        return 1/sqrt(1+(e*r)^2)
    end

    fs = [f0, f1, f2, f3]
    nc = 100
    n = 10
    xc = rand(nc)
    yc = rand(nc)
    zc = rand(nc)
    x = rand(n)
    y = rand(n)
    z = rand(n)
    c = rand(nc)
    d = rand(4)
    e = rand(nc)

    for kind = 1:4
        out = zeros(n)
        for i = 1:n 
            r = @. sqrt((x[i] - xc)^2 + (y[i] - yc)^2 + (z[i] - zc)^2)
            out[i] = sum(c .* fs[kind].(r, e)) + d[1] + d[2] * x[i] + d[3] * y[i] + d[4] * z[i]
        end

        rbf = RBF3D(xc, yc, zc, c = c, eps = e, d = d, kind = kind - 1)
        o = rbf(x,y,z)

        o_ = run(sess, o)

        @test maximum(abs.(out .- o_)) < 1e-8
    end

    for kind = 1:4
        out = zeros(n)
        for i = 1:n 
            r = @. sqrt((x[i] - xc)^2 + (y[i] - yc)^2 + (z[i] - zc)^2)
            out[i] = sum(c .* fs[kind].(r, e)) + d[1]
        end

        rbf = RBF3D(xc, yc, zc, c = c, eps = e, d = d[1:1], kind = kind - 1)
        o = rbf(x,y,z)

        o_ = run(sess, o)

        @test maximum(abs.(out .- o_)) < 1e-8
    end

    for kind = 1:4
        out = zeros(n)
        for i = 1:n 
            r = @. sqrt((x[i] - xc)^2 + (y[i] - yc)^2 + (z[i] - zc)^2)
            out[i] = sum(c .* fs[kind].(r, e))
        end

        rbf = RBF3D(xc, yc, zc, c = c, eps = e, kind = kind - 1)
        o = rbf(x,y,z)

        o_ = run(sess, o)

        @test maximum(abs.(out .- o_)) < 1e-8
    end
end