using MPI 
using ADCME

function basis(a)
    basis_ = load_op_and_grad("./Basis/build/libBasis","basis")
    a = convert_to_tensor(Any[a], [Float64]); a = a[1]
    basis_(a)
end

function m_sum(a)
    m_sum_ = load_op_and_grad("./Sum/build/libMSum","m_sum")
    a = convert_to_tensor(Any[a], [Float64]); a = a[1]
    m_sum_(a)
end



MPI.Init()

a = constant(2.0)
b = basis(a)
c = m_sum(b)
g = gradients(c, a)

sess = Session(); init(sess)
result = run(sess, c)
grad = run(sess, g)


if MPI.Comm_rank(MPI.COMM_WORLD)==0
    @info result, grad
end


