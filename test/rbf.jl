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