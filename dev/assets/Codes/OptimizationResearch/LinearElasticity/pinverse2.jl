using Revise
using AdFem
using PyPlot
using LinearAlgebra
using Statistics
using MAT 
function f(x, y)
        1/(1+x^2) + x * y + y^2
    end
    
mmesh = Mesh(50, 50, 1/50, degree=2)

left = bcnode((x,y)->x<1e-5, mmesh)
right = bcedge((x1,y1,x2,y2)->(x1>0.049-1e-5) && (x2>0.049-1e-5), mmesh)

t1 = eval_f_on_boundary_edge((x,y)->1.0e-4, right, mmesh)
t2 = eval_f_on_boundary_edge((x,y)->0.0, right, mmesh)
rhs = compute_fem_traction_term(t1, t2, right, mmesh)

nu = 0.3
x = gauss_nodes(mmesh)
E = abs(fc(x, [20, 20, 20, 1])|>squeeze)
# E = constant(eval_f_on_gauss_pts(f, mmesh))

D = compute_plane_stress_matrix(E, nu*ones(get_ngauss(mmesh)))
K = compute_fem_stiffness_matrix(D, mmesh)

bdval = [eval_f_on_boundary_node((x,y)->0.0, left, mmesh);
        eval_f_on_boundary_node((x,y)->0.0, left, mmesh)]
DOF = [left;left .+ mmesh.ndof]
K, rhs = impose_Dirichlet_boundary_conditions(K, rhs, DOF, bdval)
u = K\rhs 
U = matread("fenics/data2.mat")["u"]

using Random; Random.seed!(233)
# idx = rand(1:mmesh.ndof, 1000)
idx = 1:mmesh.ndof
idx = [idx; idx .+ mmesh.ndof]

loss = sum((U[idx] - u[idx])^2) * 1e10
sess = Session(); init(sess)

_loss = Float64[]
cb = (vs, iter, loss)->begin 
    global _loss
    push!(_loss, loss)
    printstyled("[#iter $iter] loss=$loss\n", color=:green)
    nu = vs[1]
    close("all")
    visualize_scalar_on_gauss_points(nu, mmesh)
    if mod(iter, 10)==1
        matwrite("fenics/data2-inverse$iter.mat", Dict("iter"=>iter,"loss"=>_loss, "E"=>nu))
        savefig("fenics/inverse2_nn$iter.png")
    end
end

# run(sess, loss)
loss_ = BFGS!(sess, loss, vars = [E], callback = cb)

