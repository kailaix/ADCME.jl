using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)
include("poisson.jl")
function poisson(u,up,down,left,right,f,h)
    poisson_ = load_op_and_grad("./build/libPoisson","poisson")
    u,up,down,left,right,f,h = convert_to_tensor(Any[u,up,down,left,right,f,h], [Float64,Float64,Float64,Float64,Float64,Float64,Float64])
    poisson_(u,up,down,left,right,f,h)
end

mpi_init()
r = mpi_rank()
s = mpi_size()
M = 2
N = 2
@assert s==M*N 
I = div(r, M)
J = r%M

m = 10
n = 10
h = 1/(m+1)
f = constant(rand(n, m))
u = constant(rand(n, m))

up_ = constant(zeros(m))
down_ = constant(zeros(m))
left_ = constant(zeros(n))
right_ = constant(zeros(n))


up = constant(zeros(m))
down = constant(zeros(m))
left = constant(zeros(n))
right = constant(zeros(n))

(I>0) && (global up = u[1,:])
(I<N-1) && (global down = u[end,:])
(J>0) && (global left = u[:,1])
(J<M-1) && (global right = u[:,end])

# up_ = mpi_sendrecv(down, 2, 0)

if I>0
    @info "Sending to $((I-1)*M+J) I am $r"
    global op1 = mpi_send(up, (I-1)*M+J)
end

if I<N-1
    @info "Receiving from $((I+1)*M+J) I am $r"
    global down_ = mpi_recv(up, (I+1)*M+J)
end

if I<N-1
    global op2 = mpi_send(down, (I+1)*M+J)
    global down_ = bind(down_, op2)
end

if I>0
    global up_ = mpi_recv(op1, (I-1)*M+J)
end 

if J<M-1
    op3 = mpi_send(right, I*M+J+1)
    global left_ = bind(left_, op3)
end

if J>0
    global left_ = mpi_recv(left ,I*M+J-1)
end

if J>0
    op4 = mpi_send(left, I*M+J-1)
    global right_ = bind(right_, op4)
end

if J<M-1
    global right_ = mpi_recv(op3 ,I*M+J+1)
end



# (I>0) && (global up_  = mpi_sendrecv(down, r, (I-1)*M+J))
# (I<N-1) && (global down_  = mpi_sendrecv(up, r, (I+1)*M+J))
# (J<M-1) && (global right_  = mpi_sendrecv(left, r, I*M+J+1))
# (J>0) && (global left_  = mpi_sendrecv(right, r, I*M+J-1))

u = poisson(u, up_, down_, left_, right_, f, h)

sess = Session(); init(sess)
@info r, run(sess, u)

mpi_finalize()