export Poisson, make_library, MPIBlockDims, PoissonSolver
function Poisson(u,f,h)
    if MPI.Comm_size(comm)!=blockM*blockN 
        error("Total number of sizes must be $(blockM*blockN), current = $(MPI.Comm_size(comm))")
    end
    poisson_op_ = load_op_and_grad("$(@__DIR__)/../deps/Poisson/build/libPoissonOp","poisson_op")
    u,f,h = convert_to_tensor(Any[u,f,h], [Float64,Float64,Float64])
    poisson_op_(u,f,h)
end

function PoissonSolver(u0, f, h; max_iter = 20)
    u, f, h = convert_to_tensor([u0, f, h], [Float64, Float64, Float64])
    u_arr = TensorArray(max_iter+1)
    u_arr = write(u_arr, 1, u)
    function condition(i, u_arr)
        i<=max_iter+1
    end
    function body(i, u_arr)
        u = read(u_arr,i-1)
        u_new = Poisson(u, f, h)
        i+1, write(u_arr, i, u_new)
    end
    i = constant(2, dtype = Int32)
    _, u = while_loop(condition, body, [i, u_arr])
    u = stack(u)
    return u[max_iter+1]
end

function MPIBlockDims(M=1, N=1; force=false)
    cnt = String(read("$(@__DIR__)/../deps/Poisson/CMakeLists.txt"))
    if force || !occursin("-DPOISSON_M=$M -DPOISSON_N=$N", cnt)
        make_library(M, N)
    end
    global blockM, blockN
    blockM = M
    blockN = N 
    nothing 
end

function make_library(M = 1, N = 1)
    cnt = String(read("$(@__DIR__)/../deps/Poisson/CMakeListsCopy.txt"))
    cnt = replace(cnt, "-DPOISSON_M=1 -DPOISSON_N=1"=>"-DPOISSON_M=$M -DPOISSON_N=$N")
    open("$(@__DIR__)/../deps/Poisson/CMakeLists.txt", "w") do io
        write(io, cnt)
    end
    ADCME.make_library("$(@__DIR__)/../deps/Poisson")
    global blockM, blockN
    blockM = M
    blockN = N 
    nothing 
end