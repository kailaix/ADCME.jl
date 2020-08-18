include("ops.jl")
using Test

# TODO: specify your input parameters
indices = [
    0 0
    0 2
    0 4
    1 3
    1 4
    4 2
]
values = [1.0;2.0;3.0;4.0;5.0;6.0]
ilower = 10
iupper = 15
rows, ncols, cols, out =  mpi_create_matrix(indices,values,ilower,iupper)
sess = Session(); init(sess)
@test run(sess, rows) == Int32[0, 1, 4] .+ ilower
@test run(sess, cols) == Int32[0, 2, 4, 3, 4, 2]
@test run(sess, ncols) == Int32[3, 2, 1]
@test run(sess, out) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(x)
    return sum(mpi_create_matrix(indices,x,ilower,iupper)[end]^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(6))
v_ = rand(6)
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
savefig("test.png")