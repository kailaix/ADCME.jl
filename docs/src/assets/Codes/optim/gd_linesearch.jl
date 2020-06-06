using LineSearches
using Revise
using ADCME
using NNFEM
using LinearAlgebra
using PyPlot
using JLD2 
using Statistics

# ls = HagerZhang()
ls = BackTracking()
# ls = MoreThuente()
# ls = Static()


include("compute_loss.jl")

sess = Session(); init(sess)
uo = UnconstrainedOptimizer(sess, loss)
x0 = getInit(uo)
for i = 1:1000
    global x0 
    f, df = getLossAndGrad(uo, x0)
    setSearchDirection!(uo, x0, -df)
    α, fx = linesearch(uo, f, df, ls, 200.0)
    x0 -= α*df 
    @info i, α, fx 
end