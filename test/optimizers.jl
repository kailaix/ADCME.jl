using Random 

function _benchmark_opt(tfopt, adcmeopt)
    reset_default_graph()
    n = 100
    Random.seed!(233)
    θ = Variable(ae_init([2,10,10,2]))
    y = fc(rand(10,2), [10,10,2], θ)
    loss = sum(y^2)
    opt = tfopt().minimize(loss)

    global sess = Session(); init(sess)
    qs = zeros(10)
    for i = 1:10
        qs[i] = run(sess, loss)
        run(sess, opt)
    end

# ------------------ ADCME ----------------------
    init(sess)
    opt = adcmeopt()
    uo = UnconstrainedOptimizer(sess, loss, vars=[θ])
    x0 = getInit(uo)
    fs = zeros(10)
    for i = 1:10
        fs[i], df = getLossAndGrad(uo, x0)
        Δ = apply!(opt, x0, df)
        x0 -= Δ 
    end
    flag =  maximum(abs.(fs-qs)./abs.(qs))<1e-2
    @info maximum(abs.(fs-qs)./abs.(qs))
    println(fs)
    println(qs)
    return flag
end


function _print_opt(adcmeopt)
    reset_default_graph()
    n = 100
    Random.seed!(233)
    θ = Variable(ae_init([2,10,10,2]))
    y = fc(rand(10,2), [10,10,2], θ)
    loss = sum(y^2)
    global sess = Session(); init(sess)
    opt = adcmeopt()
    uo = UnconstrainedOptimizer(sess, loss, vars=[θ])
    x0 = getInit(uo)
    fs = zeros(10)
    for i = 1:10
        fs[i], df = getLossAndGrad(uo, x0)
        Δ = apply!(opt, x0, df)
        x0 -= Δ 
    end
    println("$adcmeopt: ")
    println(fs)
end

@testset "Optimizers" begin 
    @test _benchmark_opt(GradientDescentOptimizer, Descent)
    @test _benchmark_opt(AdamOptimizer, ADAM)
    @test _benchmark_opt(AdagradOptimizer, ADAGrad)
    @test _benchmark_opt(AdadeltaOptimizer, ADADelta)
    _print_opt(RMSProp)
    _print_opt(AMSGrad)
    _print_opt(NADAM)
    _print_opt(Momentum)
    _print_opt(Nesterov)
    _print_opt(RADAM)
    _print_opt(AdaMax)
end 
