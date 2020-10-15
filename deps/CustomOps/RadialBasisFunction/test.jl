using Revise
using ADCME
using LinearAlgebra

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
        d = sqrt((x[i]-xc[j])^2 + (y[i]-yc[j])^2)
        O[i] += c[j] * exp(-(e[j]*d)^2)
    end
    O[i] += d[1] + d[2] * x[i] + d[3] * y[i]
end

sess = Session(); init(sess)
@test norm(run(sess, o)-O)<1e-5