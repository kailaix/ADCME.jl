using ADCME
using PyCall
using PyPlot
using DelimitedFiles
np = pyimport_conda("numpy", "numpy")

# load operators
if Sys.islinux()
py"""
import tensorflow as tf
libDirichletBD = tf.load_op_library('DirichletBD/build/libDirichletBD.so')
@tf.custom_gradient
def dirichlet_bd(ii,jj,dof,vv):
    uu = libDirichletBD.dirichlet_bd(ii,jj,dof,vv)
    def grad(dy):
        return libDirichletBD.dirichlet_bd_grad(dy, uu, ii,jj,dof,vv)
    return uu, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libDirichletBD = tf.load_op_library('DirichletBD/build/libDirichletBD.dylib')
@tf.custom_gradient
def dirichlet_bd(ii,jj,dof,vv):
    uu = libDirichletBD.dirichlet_bd(ii,jj,dof,vv)
    def grad(dy):
        return libDirichletBD.dirichlet_bd_grad(dy, uu, ii,jj,dof,vv)
    return uu, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libDirichletBD = tf.load_op_library('DirichletBD/build/libDirichletBD.dll')
@tf.custom_gradient
def dirichlet_bd(ii,jj,dof,vv):
    uu = libDirichletBD.dirichlet_bd(ii,jj,dof,vv)
    def grad(dy):
        return libDirichletBD.dirichlet_bd_grad(dy, uu, ii,jj,dof,vv)
    return uu, grad
"""
end

dirichlet_bd = py"dirichlet_bd"


if Sys.islinux()
py"""
import tensorflow as tf
libSparseSolver = tf.load_op_library('SparseSolver/build/libSparseSolver.so')
@tf.custom_gradient
def sparse_solver(ii,jj,vv,kk,ff,d):
    u = libSparseSolver.sparse_solver(ii,jj,vv,kk,ff,d)
    def grad(dy):
        return libSparseSolver.sparse_solver_grad(dy, u, ii,jj,vv,kk,ff,d)
    return u, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libSparseSolver = tf.load_op_library('SparseSolver/build/libSparseSolver.dylib')
@tf.custom_gradient
def sparse_solver(ii,jj,vv,kk,ff,d):
    u = libSparseSolver.sparse_solver(ii,jj,vv,kk,ff,d)
    def grad(dy):
        return libSparseSolver.sparse_solver_grad(dy, u, ii,jj,vv,kk,ff,d)
    return u, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libSparseSolver = tf.load_op_library('SparseSolver/build/libSparseSolver.dll')
@tf.custom_gradient
def sparse_solver(ii,jj,vv,kk,ff,d):
    u = libSparseSolver.sparse_solver(ii,jj,vv,kk,ff,d)
    def grad(dy):
        return libSparseSolver.sparse_solver_grad(dy, u, ii,jj,vv,kk,ff,d)
    return u, grad
"""
end

sparse_solver = py"sparse_solver"
    
function assemble_FEM(Ds, Fs, nodes, elem)
    NT = size(elem,1)
    cond0 = (i,tai,taj,tav, tak, taf) -> i<=NT
    elem = constant(elem)
    nodes = constant(nodes)
    function body(i, tai, taj, tav, tak, taf)
        el = elem[i]
        x1, y1 = nodes[el[1]][1], nodes[el[1]][2]
        x2, y2 = nodes[el[2]][1], nodes[el[2]][2]
        x3, y3 = nodes[el[3]][1], nodes[el[3]][2]
        T = abs(0.5*x1*y2 - 0.5*x1*y3 - 0.5*x2*y1 + 0.5*x2*y3 + 0.5*x3*y1 - 0.5*x3*y2)
        D = Ds[i]; F = Fs[i]*T/3
        v = T*stack([D*((-x2 + x3)^2/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)^2 + (y2 - y3)^2/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)^2),D*((x1 - x3)*(-x2 + x3)/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)^2 + (-y1 + y3)*(y2 - y3)/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)^2),D*((-x1 + x2)*(-x2 + x3)/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)^2 + (y1 - y2)*(y2 - y3)/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)^2),D*((x1 - x3)*(-x2 + x3)/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)^2 + (-y1 + y3)*(y2 - y3)/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)^2),D*((x1 - x3)^2/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)^2 + (-y1 + y3)^2/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)^2),D*((-x1 + x2)*(x1 - x3)/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)^2 + (-y1 + y3)*(y1 - y2)/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)^2),D*((-x1 + x2)*(-x2 + x3)/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)^2 + (y1 - y2)*(y2 - y3)/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)^2),D*((-x1 + x2)*(x1 - x3)/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)^2 + (-y1 + y3)*(y1 - y2)/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)^2),D*((-x1 + x2)^2/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)^2 + (y1 - y2)^2/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)^2)])
        tav = write(tav, i, v)
        ii = vec([elem[i] elem[i] elem[i]]')
        jj = [elem[i]; elem[i]; elem[i]]
        tai = write(tai, i, ii)
        taj = write(taj, i, jj)
        tak = write(tak, i, elem[i])
        taf = write(taf, i, stack([F,F,F]))
        return i+1, tai, taj, tav, tak, taf
    end
    tai = TensorArray(NT, dtype=Int32)
    taj = TensorArray(NT, dtype=Int32)
    tak = TensorArray(NT, dtype=Int32)
    tav = TensorArray(NT)
    taf = TensorArray(NT)
    i = constant(1, dtype=Int32)
    i, tai, taj, tav, tak, taf = body(i, tai, taj, tav, tak, taf)
    _, tai, taj, tav, tak, taf = while_loop(cond0, body, [i, tai, taj, tav, tak, taf]; parallel_iterations=10)
    vec(stack(tai)[1:NT]'), vec(stack(taj)[1:NT]'), vec(stack(tav)[1:NT]'),
                        vec(stack(tak)[1:NT]'), vec(stack(taf)[1:NT]')
end

function solve_LS(bddof, ii, jj, vv, kk, ff, d)
    ii = cast(ii, Int32); jj = cast(jj, Int32); kk = cast(kk, Int32); d_= constant(d, dtype=Int32)
    vv = dirichlet_bd(ii, jj, constant(bddof, dtype=Int32), vv)
    u = sparse_solver(ii,jj,vv,kk,ff,d_)
    u.set_shape((d,))
    t = ones(Bool, d); t[bddof] .= false
    t = findall(t)
    out = constant(zeros(d))
    out = scatter_add(out, t, u[t])
end

#=
# choice 1: create mesh from distmesh
# creating geometries: you need to download distmesh first
# http://persson.berkeley.edu/distmesh/

using MATLAB
if !isdir("distmesh")
    download("http://persson.berkeley.edu/distmesh/distmesh.zip", "$(@__DIR__)/distmesh.zip")
    run(`unzip $(@__DIR__)/distmesh.zip -d $(@__DIR__)`)
    rm("$(@__DIR__)/distmesh.zip")
end
mat"""
addpath distmesh
fd=@(p) sqrt(sum(p.^2,2))-1;
[$nodes,$elem]=distmesh2d(fd,@huniform,0.2,[-1,-1;1,1],[]);
$e=boundedges($nodes,$elem)
"""
elem = Int32.(elem)
dof = Int32.(e)[:]|>unique
=#

# choice 2: load data from data folder
nodes = readdlm("$(@__DIR__)/meshdata/nodes.txt")
elem = readdlm("$(@__DIR__)/meshdata/elem.txt", '\t', Int32)
dof = readdlm("$(@__DIR__)/meshdata/dof.txt", '\t', Int32)[:]



fn = (x,y) -> 4.0
dn = (x,y) -> 1.0
un = (x,y) -> 1-x^2-y^2
el1 = elem[:,1]; el2 = elem[:,2]; el3 = elem[:,3]
center = [(nodes[el1,1]+nodes[el2,1]+nodes[el3,1])/3 (nodes[el1,2]+nodes[el2,2]+nodes[el3,2])/3]
Ds = dn.(center[:,1], center[:,2]); Fs = fn.(center[:,1], center[:,2])
Ds = constant(Ds); Fs = constant(Fs)
ii, jj, vv, kk, ff = assemble_FEM(Ds, Fs, nodes, elem)
d = size(nodes,1)
u = solve_LS(dof, ii, jj, vv, kk, ff, d)

sess = Session(); init(sess)
uval = run(sess, u)

# Visualization
close("all")
scatter3D(nodes[:,1], nodes[:,2], uval, label="numerical")
x0 = LinRange(-1.0,1.0,100)
x0,y0 = np.meshgrid(x0,x0)
z0 = un.(x0,y0)
z0[x0.^2+y0.^2 .>= 1.0] .= 0.0
plot_surface(x0,y0,z0,alpha=0.5,color="orange")
xlabel("x")
ylabel("y")
legend()

# inspect the gradients of sum(u) w.r.t. Ds 
println(run(sess, gradients(sum(u), Ds))) # a sparse tensor
println(run(sess, tf.convert_to_tensor(gradients(sum(u), Ds)))) # full tensor
