using SparseArrays


# @testset "sparse" begin
#     a = Variable(1.0)
#     sp = sparsetensor([2;1],[1;2],[2a;1.0])
#     sp_a = sparsetensor([2;1],[1;2],[2a;1.0]; row_major=false)
#     y = ones(2)
#     x1 = sparse_dense_matmul(sp, y)
#     x2 = sparse_dense_matmul(sp_a, y, adjoint_a=true)
#     init(sess)
#     run(sess, Array(sp))≈[0.0 1.0;2.0 0.0]
#     run(sess, Array(sp_a))≈[0.0 1.0;2.0 0.0]
#     run(sess, x1)≈[1.0;2.0]
#     run(sess, x2)≈[2.0;1.0]
# end