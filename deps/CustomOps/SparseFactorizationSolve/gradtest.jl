using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using SparseArrays
# using ADCMEKit
Random.seed!(233)

function sparse_factorization(A, s=99)
    ii, jj, vv = find(constant(A))
    d = size(A, 1)
    sparse_factorization_ = load_op_and_grad("./build/libfactorization","sparse_factorization")
    ii,jj,vv,d,s = convert_to_tensor([ii,jj,vv,d,s], [Int64,Int64,Float64,Int64,Int64])
    stop_gradient(sparse_factorization_(ii,jj,vv,d,s))
end

function sparse_solve(A,rhs,o)
    ii, jj, vv = find(constant(A))
    solve_ = load_op_and_grad("./build/libSolve","solve")
    rhs,ii, jj, vv,o = convert_to_tensor([rhs,ii, jj, vv,o], [Float64,Int64, Int64, Float64,Int64])
    solve_(rhs,ii, jj, vv,o)
end

function sparse_factorization_solve(A, rhs)
    o = sparse_factorization(A)
    sparse_solve(A, rhs, o)
end

# TODO: specify your input parameters
A = sprand(10,10,0.3)
rhs1 = rand(10)
rhs2 = rand(10)
u = sparse_factorization(A)

out1 = sparse_solve(A, rhs1, u)
out2 = sparse_solve(A, rhs2, u)
sess = Session(); init(sess)
u, o1, o2 = run(sess, [u, out1, out2]) 
@info u 
@show o1 - A\rhs1
@show o2 - A\rhs2


# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
ii, jj, vv = find(constant(A))
rhs = rand(10)

function scalar_function(m)
    A = SparseTensor(ii, jj, m, 10, 10)
    return sum(sparse_factorization_solve(A, rhs)^2)
end

# TODO: change `m_` and `v_` to appropriate values
k = length(vv)
m_ = constant(rand(k))
v_ = rand(k)
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


A = sprand(10, 10, 0.8)
ii, jj, vv = find(constant(A))
k = length(vv)
function while_loop_simulation(vv, rhs, ns = 10)
    A = SparseTensor(ii, jj, vv, 10, 10) + spdiag(10)*100.

    o = sparse_factorization(A)

    ta = TensorArray(ns)
    i = constant(2, dtype=Int32)
    ta = write(ta, 1, ones(10))
    function condition(i, ta)
        i<= ns
    end
    function body(i, ta)
        u = read(ta, i-1)
        res = sparse_solve(A, u + rhs, o)
        # res = u 
        ta = write(ta, i, res)
        i+1, ta 
    end
    _, out = while_loop(condition, body, [i, ta])
    sum(stack(out)^2)
end

vv_ = run(sess, vv)
pl = placeholder(vv_)

# test for vv 
pl = placeholder(rand(k))
res = while_loop_simulation(pl, rhs , 100)
gradview(sess, pl, res, rand(k))

# test for right hand side 
pl = placeholder(rand(10))
res = while_loop_simulation(vv_, pl , 100)
gradview(sess, pl, res, rand(10))
