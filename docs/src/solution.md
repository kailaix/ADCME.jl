# Solution: Inverse Modeling with ADCME

## Problem 1

**(a)**
$$
\begin{aligned}
\min_{\kappa}\ & ( u(0, t)- u_0(t))^2\\
\mathrm{s.t.}\ & \frac{\partial u(x, t)}{\partial t} = \kappa(x)\Delta u(x, t) + f(x, t), \quad t\in [0,T], x\in (0,1) \\
& -\kappa(0)\frac{\partial u(0,t)}{\partial x} = 0, t>0\\
& u(1, t) = 0, t>0\\
& u(x, 0) = 0, x\in [0,1]
\end{aligned}
$$
Other reasonable loss functions are also accepted. 



**(b)** Let $\lambda = -\kappa_i \frac{\Delta t}{\Delta x^2}$, then we have
$$
A = \begin{bmatrix}
-2\lambda_1-1 & 2\lambda_1  &  & & \\
\lambda_2 & -2\lambda_2-1 & \lambda_2 & & \\
 & \lambda_3 & -2\lambda_3 - 1 & \lambda_3 & \\
& &\ddots & & \\
& & &\ddots & \lambda_{n-1}\\
&&& \lambda_n & -2\lambda_n-1
\end{bmatrix},\quad F^k = \Delta t \begin{bmatrix}
f_1^{k+1} \\
f_2^{k+1} \\
\vdots\\
f_n^{k+1}
\end{bmatrix}
$$

**(c)** and **(d)**

```julia
using ADCME 
using PyPlot

m = 50
n = 50
dt = 1/m
dx = 1/n
F = zeros(m+1, n)
xi = LinRange(0,1,n+1)[1:end-1]
f = (x,t)->exp(-50(x-0.5)^2)
for k = 1:m+1
  t = (k-1)*dt
  F[k,:] = dt*f.(xi, t)
end

κ = constant(2.0 .+ 1.5 * xi)

# TODO: Construct `A` using `spdiag`
#= 
Hint: the following syntax might be useful 
∘ Concatenate two tensors: [o1;o2]
∘ Elementwise multiplication: o1 .* o2 
∘ Indexing: o1[1], o1[1:3], o1[3:end-1]
=#
λ = -κ*dt/dx^2
mask = ones(n-1)
mask[1] =  2.0
A = spdiag(n, -1=>λ[2:end], 0=>-2λ+1, 1=>λ[1:end-1].*mask)

function condition(i, tas...)
    i<=m+1
end

function body(i, u_arr)
    u = read(u_arr, i-1)
    # TODO: Compute u_next using u and F[i]
    rhs = u + F[i]
    u_next = A\rhs
    u_arr = write(u_arr, i, u_next)
    i+1, u_arr
end

F = constant(F)
u_arr = TensorArray(m+1)
u_arr = write(u_arr, 1, zeros(n))
i = constant(2, dtype=Int32)
_, u = while_loop(condition, body, [i, u_arr])
u = set_shape(stack(u), (m+1, n))

sess = Session(); init(sess)
u0 = run(sess, u)

plot(LinRange(0,1,m+1), u0[:,1])
xlabel("Time")
ylabel("Temperature")
grid("on")
savefig("ex1_reference.png")
```

**(e)** $a=5$, $b=2$

```julia
using ADCME 
using PyPlot
using DelimitedFiles

m = 50
n = 50
dt = 1/m
dx = 1/n
F = zeros(m+1, n)
xi = LinRange(0,1,n+1)[1:end-1]
f = (x,t)->exp(-50(x-0.5)^2)
for k = 1:m+1
  t = (k-1)*dt
  F[k,:] = dt*f.(xi, t)
end

a = Variable(5.0)
b = Variable(2.0)

# TODO: Construct κ using `a` and `b`
κ = a + b * xi 

# TODO: Construct `A` using `spdiag`
#= 
Hint: the following syntax might be useful 
∘ Concatenate two tensors: [o1;o2]
∘ Elementwise multiplication: o1 .* o2 
∘ Indexing: o1[1], o1[1:3], o1[3:end-1]
=#
λ = -κ*dt/dx^2
mask = ones(n-1)
mask[1] =  2.0
A = spdiag(n, -1=>λ[2:end], 0=>-2λ+1, 1=>λ[1:end-1].*mask)

function condition(i, tas...)
    i<=m+1
end

function body(i, u_arr)
    u = read(u_arr, i-1)
    # TODO: Compute u_next using u and F[i]
    rhs = u + F[i]
    u_next = A\rhs
    u_arr = write(u_arr, i, u_next)
    i+1, u_arr
end

F = constant(F)
u_arr = TensorArray(m+1)
u_arr = write(u_arr, 1, zeros(n))
i = constant(2, dtype=Int32)
_, u = while_loop(condition, body, [i, u_arr])
u = set_shape(stack(u), (m+1, n))


uc = readdlm("data.txt")[:]

# TODO: Formulate the loss function
loss = sum((uc-u[:,1])^2) * 1e10

sess = Session(); init(sess)
BFGS!(sess, loss)
@show run(sess, [a, b])
```

## Problem 2

**(a)** 
$$
\begin{aligned}
\min_{\kappa}\ & ( u(0.2, 0.2, t)- u_1(t))^2 + ( u(0.8, 0.8, t)- u_1(t))^2\\
\mathrm{s.t.}\ & \frac{\partial u(x,y, t)}{\partial t} = \kappa\Delta u(x,y, t) + f(x,y, t), \quad t\in (0,1), (x,y)\in [0,1]^2 \\
& u(x, y, 0) = 0, \quad (x,y) \in  \Omega\\
& u(x,y,t) = 0 ,\quad (x,y)\in \partial \Omega
\end{aligned}
$$
**(b)**

```julia
using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function heat_equation(u,f,kappa,dt,dx,m,n)
    heat_equation_ = load_op_and_grad("./build/libHeatEquation","heat_equation")
    u,f,kappa,dt,dx,m,n = convert_to_tensor([u,f,kappa,dt,dx,m,n], [Float64,Float64,Float64,Float64,Float64,Int64,Int64])
    heat_equation_(u,f,kappa,dt,dx,m,n)
end

# simulation parameters 
m = 50
n = 50
NT = 50
h = 1/m
dt = 1/NT 

x = zeros((m+1)*(n+1))
y = zeros((m+1)*(n+1))
for i = 1:m+1
    for j = 1:n+1
        idx = (j-1)*(m+1)+i 
        x[idx] = (i-1)*h 
        y[idx] = (j-1)*h 
    end
end

F = zeros(NT+1, (m+1)*(n+1))
κ = zeros((m+1)*(n+1))

# TODO: Populate values into F and κ
κ = @. 1.5 + x + 2.0*y
for i = 1:NT+1
    t = (i-1)*dt 
    F[i,:] = dt * @. (exp(-t)*exp(-50*((x-0.5)^2+(y-0.5)^2)))
end


########################### Simulation Loop ########################### 
function condition(i, u)
    i <= NT+1
end

function body(i, u_arr)
    u = read(u_arr, i-1) # temperature vector at last step 
    # TODO: Compute u_next 
    u_next = heat_equation(u,F[i],κ,dt,h,m,n)
    i+1, write(u_arr, i, u_next)
end

u_arr = TensorArray(NT+1)
u_arr = write(u_arr, 1, zeros((m+1)*(n+1)))
F = constant(F) # Must be converted to Tensor, so that you can call F[i] where i is a tensor 
i = constant(2, dtype=Int32)
_, u = while_loop(condition, body, [i, u_arr])
u = set_shape(stack(u) , (NT+1, (m+1)*(n+1))) # Reshape TensorArray to a usual tensor 

########################### Simulation Ends ############################ 

# TODO: Exact values at the center 
idx = (26-1)*(m+1) + 26
uc = u[:, idx]

sess = Session(); init(sess)

# TODO: Plot the curve of temperature at (0.5,0.5)
uc = run(sess, uc)
plot(LinRange(0, 1, NT+1), uc)
xlabel("Time")
ylabel("Temperature")
grid("on")
```

**(c)** $a=2, b=3, c=3$

```julia
using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using DelimitedFiles
Random.seed!(233)

function heat_equation(u,f,kappa,dt,dx,m,n)
    heat_equation_ = load_op_and_grad("./build/libHeatEquation","heat_equation")
    u,f,kappa,dt,dx,m,n = convert_to_tensor([u,f,kappa,dt,dx,m,n], [Float64,Float64,Float64,Float64,Float64,Int64,Int64])
    heat_equation_(u,f,kappa,dt,dx,m,n)
end

# simulation parameters 
m = 50
n = 50
NT = 50
h = 1/m
dt = 1/NT 

x = zeros((m+1)*(n+1))
y = zeros((m+1)*(n+1))
for i = 1:m+1
    for j = 1:n+1
        idx = (j-1)*(m+1)+i 
        x[idx] = (i-1)*h 
        y[idx] = (j-1)*h 
    end
end

F = zeros(NT+1, (m+1)*(n+1))

a = Variable(1.)
b = Variable(1.)
c = Variable(1.)
# pl = placeholder(ones(3))
# a, b, c = pl[1], pl[2], pl[3]
κ = a + b*x + c*y
for i = 1:NT+1
    t = (i-1)*dt 
    F[i,:] = dt * @. (exp(-t)*exp(-50*((x-0.5)^2+(y-0.5)^2)))
end


########################### Simulation Loop ########################### 
function condition(i, u)
    i <= NT+1
end

function body(i, u_arr)
    u = read(u_arr, i-1) # temperature vector at last step 
    # TODO: Compute u_next 
    u_next = heat_equation(u,F[i],κ,dt,h,m,n)
    i+1, write(u_arr, i, u_next)
end

u_arr = TensorArray(NT+1)
u_arr = write(u_arr, 1, zeros((m+1)*(n+1)))
F = constant(F) # Must be converted to Tensor, so that you can call F[i] where i is a tensor 
i = constant(2, dtype=Int32)
_, u = while_loop(condition, body, [i, u_arr])
u = set_shape(stack(u) , (NT+1, (m+1)*(n+1))) # Reshape TensorArray to a usual tensor 

########################### Simulation Ends ############################ 

# TODO: Exact values at the center 
idx1 = (11-1)*(m+1) + 11
idx2 = (41-1)*(m+1) + 41
uc = u[:, [idx1;idx2]]

uobs = readdlm("data.txt")
loss = sum((uobs-uc)^2)

sess = Session(); init(sess)

BFGS!(sess, loss)
# writedlm("data.txt", run(sess, uc))
```

