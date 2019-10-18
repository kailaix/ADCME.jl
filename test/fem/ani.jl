include("Common.jl")


function nn(strain)
    @assert length(size(strain))==2 && size(strain,2)==3
    s = strain * D 
    s - (abs(s).*s)/repeat(1+sum(s^2, dims=2),1,3)
end

# function optim_fun_and_grad(u::PyObject, loss::PyObject, θ::PyObject)
#     nr = newton_raphson((u, p)->fun(u), u, missing)
#     grads = compute_nonlinear_gradients(loss, nr.x, vars, fun)
#     return fun(nr.x)[1], grads
# end

fem = Nonlinear2D(coordinates, nodes, nn)
fun = assemble(fem)
u = placeholder(zeros(fem.sdof))
r, J = fun(u)
normr = norm(r)

# newton raphson 
top = findall(coordinates[:,2].==1.0);
topdof = [2*top;2*top .- 1];

force = compute_force(tid)
J, force = ADCME.constraints(J, force, bcdof, zero(bcval))
δ = J\(force-r)

sess = Session(); init(sess)
u0 = zeros(fem.sdof)
for i = 1:100
    global u0
    res = run(sess, normr, u=>u0)
    if res<1e-8
        @info "Newton converges"
        break
    end
    δ_ = run(sess, δ, u=>u0)
    u0 -= δ_
    u0[bcdof] = bcval
end
    
    