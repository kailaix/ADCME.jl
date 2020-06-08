@testset "ADAM" begin 
    n = 100
    x = Variable(zeros(n))
    f = sum(100((x[2:end]-x[1:end-1])^2 + (1-x[1:end-1])^2))
    adam = ADAM()
    sess = Session(); init(sess)
    uo = UnconstrainedOptimizer(sess, f, vars=[x])
    x0 = getInit(uo)

    fs = zeros(3)
    f, df = getLossAndGrad(uo, x0)
    fs[1] = f
    Δ = getSearchDirection(adam, x0, -df)
    setSearchDirection!(uo, x0, Δ)
    x0 += Δ 
    

    f, df = getLossAndGrad(uo, x0)
    fs[2] = f
    Δ = getSearchDirection(adam, x0, -df)
    setSearchDirection!(uo, x0, Δ)
    x0 += Δ 

    f, df = getLossAndGrad(uo, x0)
    fs[3] = f

    @test maximum(abs.(fs - [
        9900.0
        9880.209999091778
        9860.440346212716
    ]) ./  [
        9900.0
        9880.209999091778
        9860.440346212716
    ]) < 1e-6 # compare with TensorFlow Adam  
end 





n = 100
x = Variable(zeros(n))
f = sum(100((x[2:end]-x[1:end-1])^2 + (1-x[1:end-1])^2))
adam = ADAM()
sess = Session(); init(sess)
uo = UnconstrainedOptimizer(sess, f, vars=[x])
x0 = getInit(uo)

Profile.clear();
@profile begin 
for i = 1:1000
    global x0
    f, df = getLossAndGrad(uo, x0)
    Δ = getSearchDirection(adam, x0, -df)
    setSearchDirection!(uo, x0, Δ)
    x0 += Δ 
    # @info i, f
end

end
ProfileView.view()