using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function mpi_tensor_transpose(row,col,ncol,val,n,rank,nt)
    require_mpi()
    mpi_tensor_transpose_ = load_op_and_grad("./build/libMPITensor.so","mpi_tensor_transpose", multiple=true)
    row,col,ncol,val,n,rank,nt = convert_to_tensor(Any[row,col,ncol,val,n,rank,nt], [Int32,Int32,Int32,Float64,Int64,Int64,Int64])
    indices, vals = mpi_tensor_transpose_(row,col,ncol,val,n,rank,nt)
end

mpi_init()
row = [0;1]
col = [0;1;0;1]
ncol = [2;2]
val = [1.0;2.0;5.0;6.0]
n = 2
rank = 0
nt = 2



# TODO: specify your input parameters
u = mpi_tensor_transpose(row,col,ncol,val,n,rank,nt)
sess = Session(); init(sess)
@show run(sess, u)

# uncomment it for testing gradients
error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    return sum(mpi_tensor_transpose(row,col,ncol,val,n,rank,nt)^2)
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
