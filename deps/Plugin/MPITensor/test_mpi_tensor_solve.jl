include("ops.jl")
using Test

mpi_init()
A = Float64[1 0 2 0 3
     0 0 1 2 3
     0 3 4 1 0
     0 4 2 1 0
     1 1 2 0 1]
A = A * A'
B = SparseTensor(A)
rhs = rand(5)
ilower = 0
iupper = 4
solver = "GMRES"
printlevel = 2
rows, ncols, cols, out =  mpi_create_matrix(B.o.indices,B.o.values,ilower,iupper)
# TODO: specify your input parameters

# error()
u = mpi_tensor_solve(rows,ncols,cols,out,rhs,ilower,iupper,solver,printlevel)
sess = Session(); init(sess)
u_out = run(sess, u)
u_ref = A\rhs

@show u_out - u_ref

# uncomment it for testing gradients
error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    return sum(mpi_tensor_solve(rows,ncols,cols,values,rhs,ilower,iupper,solver,printlevel)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(10,20))
v_ = rand(10,20)
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
