# Topological Optimization 


In this section, we present the ADCME implementation of a structural topology optimization problem. The optimization problem can be mathematically described as 

$$\begin{aligned}\min_x &\; l(x, u) \\ \text{s.t.} &\; V(x) = fV_0(x) \\ &\; F(x, u) = 0 \\ &\; 0<x_{\min} < x \leq  1 \end{aligned}$$

Here $x$ is a design variable, such as density in each element. $u$ is the state variable, such as the displacement vector. $F(x, u) = 0$ is the governing equation. $V(x)$ is the total volumn and $f$ is the prescribed volumn fraction. $x_{\min}$ is the lower bound for the design variable. 

Specifically, we consider a static linear elasticity load problem, where the governing equation is discretized to a linear system 

$$K(x) U - F = 0$$

Here $U$ is the discretized solution for $u$, $F$ is the load vector, $K(x)$ is the stiffness matrix. The discretized loss function $L$ is the strain energy, which has the form 

$$L(x, U) = U^T K(x) U = F^T K(x)^{-1} F$$

The original optimization problem becomes a constrained optimization problem. The following code is used for forward computation

```julia
using AdFem 

m = 32
n = 20 
h = 1.0
fracvol = 0.4
p = 3.0
x = Variable(fracvol*ones(m*n))
ρ = reshape(repeat(x^p, 1, 4), (-1,1))
ke = compute_plane_stress_matrix(1.0,0.3)
ρ = reshape(ρ * reshape(ke, 1, 9), (-1,3,3))
K = compute_fem_stiffness_matrix(ρ, m, n, h)

bdedge = bcedge("right", m, n, h)
t1 = zeros(size(bdedge,1))
t2 = zeros(size(bdedge, 1))
t2[end] = 0.0001
F = compute_fem_traction_term([t1 t2],bdedge, m, n, h)

bdnode = bcnode("left", m, n, h)

K_, F_ = impose_Dirichlet_boundary_conditions(K, F, [bdnode; bdnode .+ (m+1)*(n+1)], zeros(2length(bdnode)))
sol = K_\F_
```

Here shows the initial guess for $x$:

```julia
using PyPlot 
sess = Session(); init(sess)
SOL = run(sess, sol)
visualize_displacement(reshape(SOL, 1, :), m, n, h)
savefig("init_opt.png")
```

We will use the Ipopt optimizer to solve the constraint optimization problem. The following code 

```julia
import Ipopt

loss = sum(sol'*K*sol)

function eval_g(x, g)
    g[1] = sum(x) - fracvol*m*n
end
function eval_jac_g(x, mode, rows, cols, values)
  if mode == :Structure
    for i = 1:length(x)
        rows[i] = 1; cols[i] = i
    end
  else
    for i = 1:length(x)
        values[i] = 1.0
    end
  end
end

function opt(f, g, fg, x0, kwargs...)
    prob = Ipopt.createProblem(m*n, 1e-6*ones(m*n), ones(m*n), 1, zeros(1), zeros(1), m*n, 0,
                     f, eval_g, (x,G)->g(G, x), eval_jac_g, nothing)
    prob.x = x0 
    Ipopt.addOption(prob, "hessian_approximation", "limited-memory")
    
    status = Ipopt.solveProblem(prob)
    println(Ipopt.ApplicationReturnStatus[status])
    Ipopt.freeProblem(prob)
    nothing
end

sess = Session(); init(sess)
losses = Optimize!(sess, loss, optimizer = opt)

visualize_scalar_on_fvm_points(run(sess, x).^p, m, n, h, vmin = 0, vmax = 1)
```


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/beam.png?raw=true)
