# if Sys.islinux()
#     @testset "mpi_SparseTensor" begin
#         A = sprand(10,10,0.3)
#         B = mpi_SparseTensor(A)
#         @test run(sess, B) ≈ A

#         B = mpi_SparseTensor(A+5I)
#         ADCME.options.mpi.solver = "GMRES"
#         g = rand(10)
#         @test run(sess, B\g) ≈ (A+5I)\g
#     end
# end