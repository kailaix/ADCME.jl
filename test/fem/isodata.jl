
include("Common.jl")

fid = 1

E = 2.1*10^11;            
nu = 0.3;                       
D = constitutive_relation(E, nu)

force = compute_force(fid)
fem = LinearElasticity2D(coordinates, nodes, D)
A, b = computeLinearSystem(fem, force, bcdof, bcval)
u = A\b
# error()
sess = Session(); init(sess)
displacement = run(sess, u)
writedlm("data/iso$fid.txt", displacement)

xx = Float64[]
yy = Float64[]
zz = Float64[]
ww = Float64[]
for i = 1:nnode
    push!(xx, coordinates[i,1])
    push!(yy, coordinates[i,2])
    push!(zz, displacement[(i-1)*ndof+1])
    push!(ww, displacement[(i-1)*ndof+2])
end
scatter(xx, yy)
scatter(zz+xx, ww+yy)