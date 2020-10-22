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

ν = 0.3 * ones(get_ngauss(mmesh))
E = eval_f_on_gauss_pts(f, mmesh)
D = compute_plane_stress_matrix(E, ν)
K = compute_fem_stiffness_matrix(D, mmesh)

bdval = [eval_f_on_boundary_node((x,y)->0.0, left, mmesh);
        eval_f_on_boundary_node((x,y)->0.0, left, mmesh)]
DOF = [left;left .+ mmesh.ndof]
K, rhs = impose_Dirichlet_boundary_conditions(K, rhs, DOF, bdval)
u = K\rhs 
sess = Session(); init(sess)
S = run(sess, u)

matwrite("fenics/data2.mat", Dict("u"=>S, "E"=>E))

close("all")
visualize_scalar_on_gauss_points(E, mmesh)
savefig("fenics/E2.png")

close("all")
figure(figsize=(20, 5))
subplot(131)
visualize_scalar_on_fem_points(S[1:mmesh.nnode], mmesh)
title("x displacement")
subplot(132)
visualize_scalar_on_fem_points(S[1+mmesh.ndof:mmesh.nnode + mmesh.ndof], mmesh)
title("y displacement")
subplot(133)
Dval = run(sess, D)
visualize_von_mises_stress(Dval, S, mmesh)
title("von Mises Stress")
savefig("fenics/fwd2.png")