using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using SparseArrays 
Random.seed!(233)

A = sprand(10,10,0.3)
bd = [1;3;5]
B = zero_out_row(A, bd)
sess = Session(); init(sess)
A[bd,:] .= 0
@show run(sess, B) - A

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
A = sprand(10,10,0.3)
bd = [1;3;5]
B = constant(A)

function scalar_function(m)
    C = RawSparseTensor(B.o.indices, m, size(B)...)
    return sum(zero_out_row(C, bd))^2
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(nnz(A)))
v_ = rand(nnz(A))
y_ = scalar_function(m_)
dy_ = gradients(y_, m_)
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 1 / 10^(1:5)

for i = 1:5
    g_ = gs_[i]
    ms_[i] = m_ + g_*v_
    ys_[i] = scalar_function(ms_[i])
    s_[i] = ys_[i] - y_
    w_[i] = s_[i] - g_*sum(v_.*dy_)
end

close("all")
sess = Session(); init(sess)
sval_ = run(sess, s_)
wval_ = run(sess, w_)
close("all")
loglog(gs_, abs.(sval_), "*-", label="finite difference")
loglog(gs_, abs.(wval_), "+-", label="automatic differentiation")
loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")
savefig("gradtest.png")