include("Common.jl")

E = Variable(1.0)*1e11
nu = Variable(0.1)
D = constitutive_relation(E, nu)

fem = LinearElasticity2D(coordinates, nodes, D)
top = findall(coordinates[:,2].==1.0);
topdof = [2*top;2*top .- 1];

loss = constant(0.0)
for fid = 1:4
    uval = readdlm("data/iso$fid.txt")
    force = compute_force(fid)
    A, b = computeLinearSystem(fem, force, bcdof, bcval)
    u = A\b
    global loss += sum((uval[topdof]-u[topdof])^2)
end

sess = Session(); init(sess)
@show run(sess, loss)
BFGS!(sess, loss, 200)