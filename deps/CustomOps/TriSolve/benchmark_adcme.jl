using ADCME 
using PyPlot
using ForwardDiff
# ADCME
function myloss_adcme(b; n=101) 
    h = 1/(n-1)
    x = LinRange(0,1,n)[2:end-1]
    f = @. (4*(2 + x - x^2))

    u = trisolve(-b/h^2 * ones(n-3), 2b/h^2+1 * ones(n-2), -b/h^2 * ones(n-3), f)
    ue = u[div(n+1,2)] # extract values at x=0.5

    return (ue-1.0)^2
end

function benchmark_adcme(n)
    b = Variable(1.0)
    loss = myloss_adcme(b; n = n)
    g = gradients(loss, b)
    sess = Session(); init(sess)
    run(sess, g)
    ts_fwd = 0.0
    ts_bwd = 0.0
    for i = 1:11
        init(sess)
        d1 = @timed run(sess, loss)
        (i>1) && (ts_fwd += d1[2])
        init(sess)
        d2 = @timed run(sess, g)  
        (i>1) && (ts_bwd += d2[2])
        @info "adcme", n, d1[2], d2[2]
    end
    ts_fwd /= 10
    ts_bwd /= 10
    return ts_fwd, ts_bwd
end

# ForwardDiff
function trisolve!(a::T, b::T, c::T, D::AbstractVector, X::AbstractVector) where T
    N = length(X)
    D = copy(D)
    B = zeros(T, N)
    @inbounds B[1] = b
    @inbounds for i = 2:N
        w = a / B[i-1]
        B[i] = b - w * c
        D[i] = D[i] - w * D[i - 1]
    end
    @inbounds X[N] = D[N] / B[N]
    @inbounds for i = N-1:-1:1
        X[i] = (D[i] - c * X[i + 1]) / B[i]
    end
    return X
end

function myloss(b::T; n=101) where T
    h = 1/(n-1)
    x = LinRange(0,1,n)[2:end-1]
    f = @. T(4*(2 + x - x^2))

    u = trisolve!(-b/h^2, 2b/h^2+1, -b/h^2, f, zeros(T, n-2))
    ue = u[div(n+1,2)] # extract values at x=0.5

    return (ue-1.0)^2
end

function benchmark_forwarddiff(n)
    ts_fwd = 0.0
    ts_bwd = 0.0
    for i = 1:10
        d1 = @timed myloss(10.0, n = n)
        ts_fwd += d1[2]
        d2 = @timed myloss(ForwardDiff.Dual(10.0, 1.0),  n = n)
        ts_bwd += d2[2]
        @info "forwarddiff", n, d1[2], d2[2]
    end
    ts_fwd /= 10
    ts_bwd /= 10
    ts_fwd, ts_bwd
end

tsf, tsb = zeros(3,6), zeros(3,6)
for (k,n) in enumerate([101, 1001,10001, 100001, 1000001, 10000001])
    tsf[1,k], tsb[1,k] = benchmark_adcme(n)
    tsf[2,k], tsb[2,k] = benchmark_forwarddiff(n)
end

figure(figsize=(10,4))
subplot(121)
plot([101, 1001,10001, 100001, 1000001, 10000001], tsf[1,:], label="ADCME")
plot([101, 1001,10001, 100001, 1000001, 10000001], tsf[2,:], label="ForwardDiff")
legend()
xlabel("\$n\$")
ylabel("Time (seconds)")
grid("on")
subplot(122)
plot([101, 1001,10001, 100001, 1000001, 10000001], tsb[1,:], label="ADCME")
plot([101, 1001,10001, 100001, 1000001, 10000001], tsb[2,:], label="ForwardDiff")
legend()
xlabel("\$n\$")
ylabel("Time (seconds)")
grid("on")
savefig("benchmark.png")