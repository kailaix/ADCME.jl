using LinearAlgebra
using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using JLD2
Random.seed!(233)

function poisson_jl(u0, up, down, left, right, f, h)
    u = copy(u0)
    n, m = size(u)
    U = zeros(n+2, m+2)
    U[2:end-1, 2:end-1] = u 
    U[2:end-1,1] = left
    U[2:end-1,end] = right
    U[1,2:end-1] = up 
    U[end,2:end-1] = down 
    for i = 1:n 
        for j = 1:m 
            u[i, j] = (U[i+1,j+2] + U[i+2,j+1] + U[i+1,j] + U[i,j+1] - h*h*f[i,j])/4
        end
    end
    u 
end

function data_exchange(left,right,up,down)
    data_exchange_ = load_op_and_grad("./build/libPoisson","data_exchange", multiple=true)
    left,right,up,down,mblock,nblock = convert_to_tensor(Any[left,right,up,down,M, N], [Float64,Float64,Float64,Float64,Int64,Int64])
    data_exchange_(left,right,up,down,mblock,nblock)
end

function poisson(u,up,down,left,right,f,h)
    poisson_ = load_op_and_grad("./build/libPoisson","poisson")
    u,up,down,left,right,f,h = convert_to_tensor(Any[u,up,down,left,right,f,h], [Float64,Float64,Float64,Float64,Float64,Float64,Float64])
    poisson_(u,up,down,left,right,f,h)
end


function update_u(u, f)
    local op1, op2, op3, op4
    r = mpi_rank()
    s = mpi_size()
    @assert s==M*N 
    I = div(r, M)
    J = r%M

    up_ = constant(zeros(m))
    down_ = constant(zeros(m))
    left_ = constant(zeros(n))
    right_ = constant(zeros(n))


    up = constant(zeros(m))
    down = constant(zeros(m))
    left = constant(zeros(n))
    right = constant(zeros(n))

    (I>0) && (up = u[1,:])
    (I<N-1) && (down = u[end,:])
    (J>0) && (left = u[:,1])
    (J<M-1) && (right = u[:,end])

    left_, right_, up_, down_ = data_exchange(left, right, up, down)

# # down → up
#     if I>0
#         op1 = mpi_send(up, (I-1)*M+J)
#     end

#     if I<N-1
#         down_ = mpi_recv(up, (I+1)*M+J)
#     end

# # up → down
#     if I<N-1
#         op2 = mpi_send(down, (I+1)*M+J)
#         u = bind(u, op2)
#     end

#     if I>0
#         up_ = mpi_recv(op1, (I-1)*M+J)
#     end 

# # left → right
#     if J<M-1
#         op3 = mpi_send(right, I*M+J+1)
#     end

#     if J>0
#         left_ = mpi_recv(left ,I*M+J-1)
#     end

# # right → left 
#     if J>0
#         op4 = mpi_send(left, I*M+J-1)
#         u = bind(u, op4)
#     end

#     if J<M-1
#         right_ = mpi_recv(op3 ,I*M+J+1)
#     end

    

    u = poisson(u, up_, down_, left_, right_, f, h)
end

function poisson_solver(f, NT=10)
    function condition(i, u_arr)
        i<=NT
    end
    function body(i, u_arr)
        u = read(u_arr, i)
        u_new = update_u(u, f)
        op = tf.print(r, i)
        u_new = bind(u_new, op)
        i+1, write(u_arr, i+1, u_new)
    end
    i = constant(1, dtype =Int32)
    u_arr = TensorArray(NT+1)
    u_arr = write(u_arr, 1, zeros(n, m))
    _, u = while_loop(condition, body, [i, u_arr])
    reshape(stack(u), (NT+1, n, m))
end