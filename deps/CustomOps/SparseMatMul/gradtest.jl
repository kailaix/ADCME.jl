using ADCME
using PyCall
using LinearAlgebra
using SparseArrays
using PyPlot
using Random
using Test
Random.seed!(233)

sparse_mat_mul = load_op("build/libSparseMatMul", "sparse_sparse_mat_mul")
diag_sparse_mat_mul = load_op("build/libSparseMatMul", "diag_sparse_mat_mul")
sparse_diag_mat_mul = load_op("build/libSparseMatMul", "sparse_diag_mat_mul")

function Base.:*(s1::SparseTensor, s2::SparseTensor)
    ii1, jj1, vv1 = find(s1)
    ii2, jj2, vv2 = find(s2)
    m, n = size(s1)
    n_, k = size(s2)
    if n!=n_
        error("IGACS: matrix size mismatch: ($m, $n) vs ($n_, $k)")
    end
    mat_mul_fn = sparse_mat_mul
    if s1._diag
        mat_mul_fn = diag_sparse_mat_mul
    elseif s2._diag
        mat_mul_fn = sparse_diag_mat_mul
    end
    ii3, jj3, vv3 = mat_mul_fn(ii1-1,jj1-1,vv1,ii2-1,jj2-1,vv2,m,n,k)
    SparseTensor(ii3, jj3, vv3, m, k)
end
# sparse_mat_mul = py"sparse_mat_mul"
################## End Load Operator ##################
A = sprand(10,5,0.3)
B = sprand(5,20,0.3)
C = A*B
CC = SparseTensor(A)*SparseTensor(B)
# TODO: specify your input parameters
sess = Session()
init(sess)
C_ = run(sess, CC)
@test C_≈C


A = spdiagm(0=>[1.;2.;3;4;5])
B = sprand(5,20,0.3)
C = A*B
CC = SparseTensor(A)*SparseTensor(B)
# TODO: specify your input parameters
sess = Session()
init(sess)
C_ = run(sess, CC)
@test C_≈C


A = sprand(10,5,0.5)
B = spdiagm(0=>[1.;2.;3;4;5])
C = A*B
CC = SparseTensor(A)*SparseTensor(B)
# TODO: specify your input parameters
sess = Session()
init(sess)
C_ = run(sess, CC)
@test C_≈C

