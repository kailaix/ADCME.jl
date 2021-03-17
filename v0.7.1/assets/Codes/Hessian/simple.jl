using ADCME 
using LinearAlgebra
using Random

npoints = parse(Int64, ARGS[1])
SEED = parse(Int64, ARGS[2])

Random.seed!(SEED)


x0 = LinRange(0, 1, 100)|>collect
y0 = sin.(π*x0)

θ = Variable(fc_init([1,20,20,20,1]))
y = fc(x0, [20,20,20,1], θ)


idx = Int64.(round.(collect(LinRange(1, 100, npoints+2))))[2:end-1]

loss = sum((y[idx] - y0[idx])^2)

H = hessian(loss, θ)

sess = Session(); init(sess)
BFGS!(sess, loss)
H0 = run(sess, H)
λ = real.(eigvals(H0))
nlambda = length(findall(λ .> λ[end] * 1e-6))

db = Database("hessian.db")
execute(db, """
CREATE TABLE IF NOT EXISTS eigvals (
    seed integer,
    npoints integer, 
    nlambda integer,
    PRIMARY KEY (seed, npoints)
)""")
execute(db, """
INSERT OR REPLACE INTO eigvals VALUES ($SEED, $npoints, $nlambda)
""")
close(db)

