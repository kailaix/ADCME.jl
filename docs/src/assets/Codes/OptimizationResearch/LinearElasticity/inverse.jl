using Revise
using AdFem
using PyPlot
using LinearAlgebra
using Statistics
using MAT 
using JLD2
function f1(x, y)
    x/0.05 + sin(10π*y)
end
function f2(x, y)
    y/0.05 + sin(10π*x) + 0.3
end


mmesh = Mesh(joinpath(PDATA, "twoholes.stl"), degree=2)

left = bcnode((x,y)->x<1e-5, mmesh)
right = bcedge((x1,y1,x2,y2)->(x1>0.049-1e-5) && (x2>0.049-1e-5), mmesh)

t1 = eval_f_on_boundary_edge((x,y)->1.0e-4, right, mmesh)
t2 = eval_f_on_boundary_edge((x,y)->0.0, right, mmesh)
rhs = compute_fem_traction_term(t1, t2, right, mmesh)

x = gauss_nodes(mmesh)

using Random; Random.seed!(233)
θ1 = Variable(ae_init([2,20, 20, 20, 1]))
θ2 = Variable(ae_init([2,20, 20, 20, 1]))
E = abs(fc(x, [20, 20, 20, 1], θ1)|>squeeze)
nu = abs(fc(x, [20, 20, 20, 1], θ2)|>squeeze)


nu0 = 0.3*eval_f_on_gauss_pts(f1, mmesh)
E0 = eval_f_on_gauss_pts(f2, mmesh)

sess = Session(); init(sess)


E_, nu_ = run(sess, [E, nu])
close("all")
figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_gauss_points(E_, mmesh)
subplot(122)
visualize_scalar_on_gauss_points(nu_, mmesh)
savefig("data/init.png")

D = compute_plane_stress_matrix(E, nu)
K = compute_fem_stiffness_matrix(D, mmesh)

bdval = [eval_f_on_boundary_node((x,y)->0.0, left, mmesh);
        eval_f_on_boundary_node((x,y)->0.0, left, mmesh)]
DOF = [left;left .+ mmesh.ndof]
K, rhs = impose_Dirichlet_boundary_conditions(K, rhs, DOF, bdval)
u = K\rhs 
U = matread("data/fwd.mat")["u"]

using Random; Random.seed!(233)
# idx = rand(1:mmesh.ndof, 1000)
idx = 1:mmesh.ndof
idx = [idx; idx .+ mmesh.ndof]

loss = sum((U[idx] - u[idx])^2) * 1e10
# sess = Session(); init(sess)

_loss = Float64[]
cb = (vs, iter, loss)->begin 
    global _loss
    push!(_loss, loss)
    printstyled("[#iter $iter] loss=$loss\n", color=:green)
    E = vs[1]
    nu = vs[2]
    close("all")
    figure(figsize=(10,4))
    subplot(121)
    visualize_scalar_on_gauss_points(E, mmesh)
    subplot(122)
    visualize_scalar_on_gauss_points(nu, mmesh)
    if mod(iter, 10)==1
        if iter==1
            make_directory("data/result$MODE")
        end
        matwrite("data/result$MODE/nn$iter.mat", Dict("iter"=>iter,"loss"=>_loss, "E"=>nu))
        savefig("data/result$MODE/nn$iter.png")
    end
end
