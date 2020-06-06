using LineSearches
using Revise
using ADCME
using NNFEM
using LinearAlgebra
using PyPlot
using JLD2 
using Statistics


include("compute_loss.jl")

# ls = HagerZhang()
ls = BackTracking()
# ls = MoreThuente()
# ls = Static()
adam = Optimizer.AMSGrad()

sess = Session(); init(sess)
uo = UnconstrainedOptimizer(sess, loss)
x0 = getInit(uo)


for i = 1:1000
    global x0 
    f, df = getLossAndGrad(uo, x0)
    Δ = getSearchDirection(adam, x0, df)
    # setSearchDirection!(uo, x0, -Δ)
    # α, fx = linesearch(uo, f, df, ls, 200.0)
    # x0 -= α*Δ 
    # @info i, α, fx, getLoss(uo, x0)
    x0 -= Δ
    @info i,  f
end