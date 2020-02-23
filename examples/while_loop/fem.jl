using ADCME, LinearAlgebra, PyCall
using DelimitedFiles
using PyPlot

# read data 
elem = readdlm("meshdata/elem.txt", Int64)
node = readdlm("meshdata/nodes.txt")
dof = readdlm("meshdata/dof.txt", Int64)[:]
elem_ = constant(elem)
ne = size(elem,1)
nv = size(node, 1)

# precompute 
localcoef = zeros(ne, 3, 3)
areas = zeros(ne)
for e = 1:ne 
    el = elem[e,:]
    x1, y1 = node[el[1],:]
    x2, y2 = node[el[2],:]
    x3, y3 = node[el[3],:]
    A = [x1 y1 1.0; x2 y2 1.0; x3 y3 1.0]
    localcoef[e,:,:] = inv(A)
    areas[e] = 0.5*abs(det(A))
end

# compute right hand side using midpoint rule 
rhs = zeros(nv)
for i = 1:ne
    el = elem[i,:]
    rhs[el] .+= 4*areas[i]/3
end

areas = constant(areas)
localcoef = constant(localcoef)
D = constant(diagm(0=>ones(2)))
# D = Variable(2.0) .* [1.0 0.0;0.0 1.0]

function body(i, tai, taj, tav)
    el = elem_[i-1]
    a = areas[i-1]
    L = localcoef[i-1]
    LocalStiff = Array{PyObject}(undef, 3, 3)
    for i = 1:3
        for j = 1:3
            LocalStiff[i,j] = a*[L[1,i] L[2,i]]*D*[L[1,j];L[2,j]]|>squeeze
        end
    end
    ii = reshape([el el el], (-1,))
    jj = reshape([el;el;el], (-1,))
    tai = write(tai, i, ii)
    taj = write(taj, i, jj)
    # op = tf.print(el)
    # i = bind(i, op)
    tav = write(tav, i, vcat(LocalStiff[:]...))
    return i+1, tai, taj, tav 
end

i = constant(2, dtype=Int32)
tai = TensorArray(ne+1, dtype=Int64)
taj = TensorArray(ne+1, dtype=Int64)
tav = TensorArray(ne+1)
tai = write(tai, 1, constant(ones(Int64,9)))
taj = write(taj, 1, constant(ones(Int64,9)))
tav = write(tav, 1, constant(zeros(9)))
_, ii, jj, vv = while_loop((i, tas...)->i<=ne+1, body, [i, tai, taj, tav])
ii = reshape(stack(ii),(-1,)); jj = reshape(stack(jj),(-1,)); vv = reshape(stack(vv),(-1,))

A = SparseTensor(ii, jj, vv, nv, nv)

ndof = [x for x in setdiff(Set(1:nv), Set(dof))]
A = scatter_update(A, dof, ndof, spzero(length(dof), length(ndof)))
A = scatter_update(A, ndof, dof, spzero(length(ndof), length(dof)))
A = scatter_update(A, dof, dof, spdiag(length(dof)))
rhs[dof] .= 0.0
sol = A\rhs 

# loss = sum((sol - (@. 1-node[:,1]^2-node[:,2]^2))^2)
sess = Session(); init(sess)
S = run(sess, sol)
close("all")
scatter3D(node[:,1], node[:,2], S, marker="^", label = "FEM")
scatter3D(node[:,1], node[:,2], (@. 1-node[:,1]^2-node[:,2]^2), marker = "+", label = "Exact")
legend()

