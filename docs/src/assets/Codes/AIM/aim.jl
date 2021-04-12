using ADCME 
using JLD2 

@load "data.jld2" z 

n = 64
noise = parse(Float64, ARGS[1])

τ = Variable(1.0)
x = placeholder(Float64, shape = [n,])
y = placeholder(Float64, shape = [n,])
w = placeholder(Float64, shape = [n,])
σ, κ = 0.08, 0.5
α = 0.5
Δt = 0.01
yhat = (
    x + κ * (τ - α*x)*Δt + σ * sqrt(x) * sqrt(Δt) * w  + 1/4*σ^2*Δt*(w^2-1)
)/(1+(1-α)*κ*Δt)

θ = Variable(fc_init([2,20,20,20,1]))
hDξ = sigmoid(squeeze(fc([x yhat], [20,20,20,1], θ)))
Dξ = sigmoid(squeeze(fc([x y], [20,20,20,1], θ)))
# LF = mean(log((1-hDξ)/hDξ))
LF = -mean(log(hDξ))
LD = -mean(log(Dξ) + log(1-hDξ))

function generate_batch_data()
    idx = rand(1:9999, n)
    abs.(z[idx] + noise * randn(n)), abs.(z[idx.+1] + noise * randn(n)), randn(n)
end


optF = RMSPropOptimizer(1e-3).minimize(LF, var_list=[τ])
optD = RMSPropOptimizer(1e-3).minimize(LD, var_list=[θ])

sess = Session(); init(sess)
x0, y0, w0 = generate_batch_data()
    fd = Dict(
        x => x0, y => y0, w => w0
    )
@info run(sess, [LF, LD], feed_dict = fd)

fiter = 1
diter = 5
l1 = []
l2 = []
τs = []

for i = 1:50000
    x0, y0, w0 = generate_batch_data()

    fd = Dict(
        x => x0, y => y0, w => w0
    )
    
    ld = 0
    for k = 1:diter
        _, ld = run(sess, [optD, LD], feed_dict = fd)
    end

    lf = 0
    for k = 1:fiter 
        _, lf = run(sess, [optF, LF], feed_dict = fd)
    end
    
    τ0 = run(sess, τ)
    
    # if length(loss)==0
    #     loss = [l1[end]]
    # end
    push!(l1, ld)
    push!(l2, lf)
    push!(τs, τ0)
    if mod(i, 100)==0
        @info i, ld, lf, τ0
    end
end

kappa = join(string.(τs), ',')
db = Database("noise.db")
execute(db, """
CREATE TABLE IF NOT EXISTS tau (
    noise real,
    kappa text
)
""")
execute(db, """
INSERT INTO tau VALUES ($noise, \"$kappa\")
""")
close(db)
# close("all")
# plot(τs)
# savefig("test$noise.png")