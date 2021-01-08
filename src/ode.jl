export ode45, rk4, αscheme, αscheme_time, 
    αscheme_atime, TR_BDF2, ExplicitNewmark

function runge_kutta_one_step(f::Function, t::PyObject, y::PyObject, Δt::PyObject, θ::Union{PyObject, Missing})
    k1 = Δt*f(t, y, θ)
    k2 = Δt*f(t+Δt/2, y+k1/2, θ)
    k3 = Δt*f(t+Δt/2, y+k2/2, θ)
    k4 = Δt*f(t+Δt, y+k3, θ)
    y = y + k1/6 + k2/3 + k3/3 + k4/6
end

function ode45_one_step(f::Function, t::PyObject, y::PyObject, h::PyObject, θ::Union{PyObject, Missing})
    k1 = h * f(t, y, θ);
    k2 = h * f(t + (1/5)*h, y + (1/5)*k1, θ);
    k3 = h * f(t + (3/10)*h, y + (3/40)*k1 + (9/40)*k2, θ);
    k4 = h * f(t + (4/5)*h, y + (44/45)*k1 + (-56/15)*k2 + (32/9)*k3, θ);
    k5 = h * f(t + (8/9)*h, y + (19372/6561)*k1 + (-25360/2187)*k2 + (64448/6561)*k3 + (-212/729)*k4, θ);
    k6 = h * f(t + h, y + (9017/3168)*k1 + (-355/33)*k2 + (46732/5247 )*k3 + (49/176)*k4 + (-5103/18656)*k5, θ);
    k7 = h * f(t + h, y + (35/384)*k1 + (500/1113)*k3 + (125/192)*k4 + (-2187/6784)*k5 + (11/84)*k6, θ);
    
    y_new = y + (35/384)*k1 + (500/1113)*k3 + (125/192)*k4 + (-2187/6784)*k5 + (11/84)*k6;
end

@doc raw"""
    runge_kutta(f::Function, T::Union{PyObject, Float64}, 
                NT::Union{PyObject,Int64}, y::Union{PyObject, Float64, Array{Float64}}, θ::Union{PyObject, Missing}=missing; method::String="rk4")

Solves 
```math
\frac{dy}{dt} = f(y, t, \theta)
```
with Runge-Kutta method. 

For example, the default solver, `RK4`, has the following numerical scheme per time step 
```math
\begin{aligned}
k_1 &= \Delta t f(t_n, y_n, \theta)\\
k_2 &= \Delta t f(t_n+\Delta t/2, y_n + k_1/2, \theta)\\
k_3 &= \Delta t f(t_n+\Delta t/2, y_n + k_2/2, \theta)\\
k_4 &= \Delta t f(t_n+\Delta t, y_n + k_3, \theta)\\
y_{n+1} &= y_n + \frac{k_1}{6} +\frac{k_2}{3} +\frac{k_3}{3} +\frac{k_4}{6}
\end{aligned}
```
"""
function runge_kutta(f::Function, T::Union{PyObject, Float64}, 
                NT::Union{PyObject,Int64}, y::Union{PyObject, Float64, Array{Float64}}, θ::Union{PyObject, Missing}=missing; method::String="rk4")

    local one_step
    if lowercase(method)=="rk4"
        one_step = runge_kutta_one_step
    elseif lowercase(method)=="rk45"
        one_step = ode45_one_step
    else
        error("Method $method not implemented yet")
    end
    y = convert_to_tensor(y)
    Δt = convert_to_tensor(T/NT)

    ta = TensorArray(NT+1) # storing y
    function condition(i, ta)
        i <= NT+1
    end
    function body(i, ta)
        y = read(ta, i-1)
        y_ = one_step(f, (cast(eltype(Δt), i)-1)*Δt, y, Δt, θ)
        ta = write(ta, i, y_)
        i+1, ta 
    end
    ta = write(ta, 1, y)
    i = constant(2, dtype=Int32)
    _, out = while_loop(condition, body, [i, ta])
    res = stack(out)
end

@doc raw"""
    rk4(y::Union{PyObject, Float64, Array{Float64}}, T::Union{PyObject, Float64}, 
                NT::Union{PyObject,Int64}, f::Function, θ::Union{PyObject, Missing}=missing)

Solves 
```math
\frac{dy}{dt} = f(y, t, \theta)
```
with Runge-Kutta (order 4) method. 
"""
rk4(args...;kwargs...) = runge_kutta(args...;method="rk4", kwargs...)

@doc raw"""
    ode45(y::Union{PyObject, Float64, Array{Float64}}, T::Union{PyObject, Float64}, 
                NT::Union{PyObject,Int64}, f::Function, θ::Union{PyObject, Missing}=missing)

Solves 
```math
\frac{dy}{dt} = f(y, t, \theta)
```
with six-stage, fifth-order, Runge-Kutta method.
"""
ode45(args...;kwargs...) = runge_kutta(args...;method="rk45", kwargs...)


@doc raw"""
    αscheme(M::Union{SparseTensor, SparseMatrixCSC}, 
        C::Union{SparseTensor, SparseMatrixCSC}, 
        K::Union{SparseTensor, SparseMatrixCSC}, 
        Force::Union{Array{Float64}, PyObject}, 
        d0::Union{Array{Float64, 1}, PyObject}, 
        v0::Union{Array{Float64, 1}, PyObject}, 
        a0::Union{Array{Float64, 1}, PyObject}, 
        Δt::Array{Float64}; 
        solve::Union{Missing, Function} = missing,
        extsolve::Union{Missing, Function} = missing, 
        ρ::Float64 = 1.0)

Generalized α-scheme. 
$$M u_{tt} + C u_{t} + K u = F$$

`Force` must be an array of size `n`×`p`, where `d0`, `v0`, and `a0` have a size `p`
`Δt` is an array (variable time step). 

The generalized α scheme solves the equation by the time stepping
```math
\begin{aligned}
\bf d_{n+1} &= \bf d_n + h\bf v_n + h^2 \left(\left(\frac{1}{2}-\beta_2 \right)\bf a_n + \beta_2 \bf a_{n+1}  \right)\\
\bf v_{n+1} &= \bf v_n + h((1-\gamma_2)\bf a_n + \gamma_2 \bf a_{n+1})\\
\bf F(t_{n+1-\alpha_{f_2}}) &= M \bf a _{n+1-\alpha_{m_2}} + C \bf v_{n+1-\alpha_{f_2}} + K \bf{d}_{n+1-\alpha_{f_2}}
\end{aligned}
```
where 
```math
\begin{aligned}
\bf d_{n+1-\alpha_{f_2}} &= (1-\alpha_{f_2})\bf d_{n+1} + \alpha_{f_2} \bf d_n\\
\bf v_{n+1-\alpha_{f_2}} &= (1-\alpha_{f_2}) \bf v_{n+1} + \alpha_{f_2} \bf v_n \\
\bf a_{n+1-\alpha_{m_2} } &= (1-\alpha_{m_2}) \bf a_{n+1} + \alpha_{m_2} \bf a_n\\
t_{n+1-\alpha_{f_2}} & = (1-\alpha_{f_2}) t_{n+1 + \alpha_{f_2}} + \alpha_{f_2}t_n
\end{aligned}
```

Here the parameters are computed using 
```math 
\begin{aligned}
\gamma_2 &= \frac{1}{2} - \alpha_{m_2} + \alpha_{f_2}\\
\beta_2 &= \frac{1}{4} (1-\alpha_{m_2}+\alpha_{f_2})^2 \\
\alpha_{m_2} &= \frac{2\rho_\infty-1}{\rho_\infty+1}\\
\alpha_{f_2} &= \frac{\rho_\infty}{\rho_\infty+1}
\end{aligned}
```

∘ `solve`: users can provide a solver function, `solve(A, rhs)` for solving `Ax = rhs`
∘ `extsolve`: similar to `solve`, but the signature has the form 
```julia
extsolve(A, rhs, i)
```
This provides the users with more control, e.g., (time-dependent) Dirichlet boundary conditions. 
See [Generalized α Scheme](https://kailaix.github.io/ADCME.jl/dev/alphascheme/) for details.

!!! note 
    In the case $u$ has a nonzero essential boundary condition $u_b$, we let $\tilde u=u-u_b$, then 
    $$M \tilde u_{tt} + C \tilde u_t + K u = F - K u_b - C \dot u_b$$
"""
function αscheme(M::Union{SparseTensor, SparseMatrixCSC}, 
                      C::Union{SparseTensor, SparseMatrixCSC}, 
                      K::Union{SparseTensor, SparseMatrixCSC}, 
                      Force::Union{Array{Float64, 2}, PyObject}, 
                      d0::Union{Array{Float64, 1}, PyObject}, 
                      v0::Union{Array{Float64, 1}, PyObject}, 
                      a0::Union{Array{Float64, 1}, PyObject}, 
                      Δt::Array{Float64, 1}; 
                      solve::Union{Missing, Function} = missing,
                      extsolve::Union{Missing, Function} = missing, 
                      ρ::Float64 = 1.0)
    if !ismissing(solve) && !ismissing(extsolve)
        error("You cannot provide `solve` and `extsolve` at the same time.")
    end
    nt = length(Δt)
    αm = (2ρ-1)/(ρ+1)
    αf = ρ/(1+ρ)
    γ = 1/2-αm+αf 
    β = 0.25*(1-αm+αf)^2
    d = length(d0)

    M = isa(M, SparseMatrixCSC) ? constant(M) : M
    C = isa(C, SparseMatrixCSC) ? constant(C) : C
    K = isa(K, SparseMatrixCSC) ? constant(K) : K
    Force, d0, v0, a0, Δt = convert_to_tensor([Force, d0, v0, a0, Δt], [Float64, Float64, Float64, Float64, Float64])

    function equ(dc, vc, ac, dt, Force, i)
        dn = dc + dt*vc + dt^2/2*(1-2β)*ac 
        vn = vc + dt*((1-γ)*ac)

        df = (1-αf)*dn + αf*dc
        vf = (1-αf)*vn + αf*vc 
        am = αm*ac 

        rhs = Force - (M*am + C*vf + K*df)
        A = (1-αm)*M + (1-αf)*C*dt*γ + (1-αf)*K*β*dt^2

        if !ismissing(solve)
            return solve(A, rhs)
        elseif !ismissing(extsolve)
            return extsolve(A, rhs, i)
        else 
            return A\rhs
        end
    end

    function condition(i, tas...)
        return i<=nt
    end
    function body(i, tas...)
        dc_arr, vc_arr, ac_arr = tas
        dc = read(dc_arr, i)
        vc = read(vc_arr, i)
        ac = read(ac_arr, i)
        y = equ(dc, vc, ac, Δt[i], Force[i], i)
        dn = dc + Δt[i]*vc + Δt[i]^2/2*((1-2β)*ac+2β*y)
        vn = vc + Δt[i]*((1-γ)*ac+γ*y)
        i+1, write(dc_arr, i+1, dn), write(vc_arr, i+1, vn), write(ac_arr, i+1, y)
    end

    dM = TensorArray(nt+1); vM = TensorArray(nt+1); aM = TensorArray(nt+1)
    dM = write(dM, 1, d0)
    vM = write(vM, 1, v0)
    aM = write(aM, 1, a0)
    i = constant(1, dtype=Int32)
    _, d, v, a = while_loop(condition, body, [i,dM, vM, aM])
    set_shape(stack(d), (nt+1, length(d0))), set_shape(stack(v), (nt+1, length(v0))), set_shape(stack(a), (nt+1, length(a0)))
end


@doc raw"""
    αscheme_time(Δt::Array{Float64}; ρ::Float64 = 1.0)

Returns the integration time $t_{i+1-\alpha_{f_2}}$ between $[t_i, t_{i+1}]$ using the alpha scheme. 
If $\Delta t$ has length $n$, the output will also have length $n$. 
"""
function αscheme_time(Δt::Array{Float64}; ρ::Float64 = 1.0)
    n = length(Δt)
    αm = (2ρ-1)/(ρ+1)
    αf = ρ/(1+ρ)
    γ = 1/2-αm+αf 
    β = 0.25*(1-αm+αf)^2
    function equ(tc, dt)
        tf1 = (1-αf)*(tc+dt) + αf*tc
        return tf1
    end

    tcf = Float64[]
    tc = 0.0
    for i = 1:n
        t1 = equ(tc, Δt[i])
        push!(tcf, t1)
        tc += Δt[i]
    end
    return tcf
end

@doc raw"""
    TR_BDF2(D0::Union{SparseTensor, SparseMatrixCSC}, 
        D1::Union{SparseTensor, SparseMatrixCSC}, 
        Δt::Float64)

Constructs a TR-BDF2 (the Trapezoidal Rule with Second Order Backward Difference Formula) handler for 
the DAE 

$$D_1 \dot y + D_0 y = f$$

The struct is a functor, which performs one step simulation 

```
(tr::TR_BDF2)(y::Union{PyObject, Array{Float64, 1}}, 
    f1::Union{PyObject, Array{Float64, 1}}, 
    f2::Union{PyObject, Array{Float64, 1}}, 
    f3::Union{PyObject, Array{Float64, 1}})
```
Here `f1`, `f2`, and `f3` correspond to the right hand side at time step $n$, $n+\frac12$, and $n+1$.

Or we can pass a batched `F` defined as a `(2NT+1) × DOF` array

```
(tr::TR_BDF2)(y0::Union{PyObject, Array{Float64, 1}}, 
    F::Union{PyObject, Array{Float64, 2}})
```

The output will be the entire solution of size `(NT+1) × DOF`.


!!! info 
    The scheme takes the following form for n = 0, 1, ...
    $$\begin{aligned} D_1(y^{n+\frac12}-y^n) = \frac12\frac{\Delta t}{2}\left(f^{n+\frac12} + f^n - D_0 \left(y^{n+\frac12} + y^n\right)\right)\\ \left(\frac{\Delta t}{2}\right)^{-1} D_1 \left(\frac32y^{n+1} - 2y^{n+\frac12} + \frac12 y^n\right) + D_0 y^{n+1} = f^{n+1}\end{aligned}$$
"""
mutable struct TR_BDF2 
    D0::Union{SparseTensor, SparseMatrixCSC} 
    D1::Union{SparseTensor, SparseMatrixCSC}  
    Δt::Float64
    _D0::Union{SparseTensor, SparseMatrixCSC} 
    _D1::Union{SparseTensor, SparseMatrixCSC}
    symbolic::Bool 
    function TR_BDF2(D0::Union{SparseTensor, SparseMatrixCSC}, 
            D1::Union{SparseTensor, SparseMatrixCSC}, 
            Δt::Float64)
        if isa(D0, SparseTensor) || isa(D1, SparseTensor)
            symbolic = true 
            D0 = constant(D0)
            D1 = constant(D1)
        else 
            symbolic = false 
        end
        @assert size(D0, 1)==size(D0,2)==size(D1,1)==size(D1,2)
        _D0 = D1 + Δt/4 * D0 
        _D1 = 1/(Δt/2) * 3/2 * D1  + D0 
        new(D0, D1, Δt, _D0, _D1, symbolic)
    end
end

"""
    constant(tr::TR_BDF2)

Converts `tr` to a symbolic solver. 
"""
function constant(tr::TR_BDF2)
    TR_BDF2(constant(tr.D0), constant(tr.D1), tr.Δt)
end

function Base.:show(io::IO, tr::TR_BDF2)
    print("""TR_BDF2 (DOF = $(size(tr.D0, 1)), Δt = $(tr.Δt)$(tr.symbolic ? ", symbolic" : ""))""")
end

function (tr::TR_BDF2)(y::Union{PyObject, Array{Float64, 1}}, 
    f1::Union{PyObject, Array{Float64, 1}}, 
    f2::Union{PyObject, Array{Float64, 1}}, 
    f3::Union{PyObject, Array{Float64, 1}})
    y, f1, f2, f3 = convert_to_tensor([y, f1, f2, f3], [Float64, Float64, Float64, Float64])
    r1 = tr.Δt/4 * (f2 + f1) - tr.Δt/4 * (tr.D0 * y) + tr.D1 * y
    yn = tr._D0\r1
    r2 = f3 + 1/(tr.Δt/2)*(tr.D1* (2*yn - 0.5*y))
    tr._D1\r2
end

function (tr::TR_BDF2)(y0::Union{PyObject, Array{Float64, 1}}, 
    F::Union{PyObject, Array{Float64, 2}})
    @assert size(F, 2)==size(tr.D0, 1) && mod(size(F, 1), 2)==1
    y0, F = convert_to_tensor([y0, F], [Float64, Float64])
    NT = size(F, 1)÷2
    y_arr = TensorArray(NT+1)
    y_arr = write(y_arr, 1, y0)
    function condition(i, y_arr)
        i<=NT
    end
    function body(i, y_arr)
        y = read(y_arr, i)
        f1, f2, f3 = F[2*i-1], F[2*i], F[2*i+1]
        yn = tr(y, f1, f2, f3)
        i+1, write(y_arr, i+1, yn)
    end
    i = constant(1, dtype = Int32)
    _, ya = while_loop(condition, body, [i, y_arr])
    set_shape(stack(ya), (NT+1, length(y0)))
end

function (tr::TR_BDF2)(y::Array{Float64, 1}, 
    f1::Array{Float64, 1}, 
    f2::Array{Float64, 1}, 
    f3::Array{Float64, 1})
    if tr.symbolic
        return tr(constant(y), f1, f2, f3)
    end
    r1 = tr.Δt/4 * (f2 + f1) - tr.Δt/4 * (tr.D0 * y) + tr.D1 * y
    yn = tr._D0\r1
    r2 = f3 + 1/(tr.Δt/2)*(tr.D1* (2*yn - 0.5*y))
    tr._D1\r2
end

function (tr::TR_BDF2)(y0::Array{Float64,1}, F::Array{Float64, 2})
    if tr.symbolic
        return tr(constant(y0), F)
    end
    @assert size(F, 2)==size(tr.D0, 1) && mod(size(F, 1), 2)==1
    y = zeros(size(F, 1)÷2+1, 2)
    y[1,:] = y0
    for i = 1:size(F, 1)÷2
        y[i+1,:] = tr(y[i,:], F[2*i-1,:], F[2*i,:], F[2*i+1,:])
    end
    y
end


# design principle: proceed one step 
# (states) u0, u1, ..., (coefficients) M0, M1, ..., (step size) Δt

@doc raw"""
    ExplicitNewmark(M::Union{SparseTensor, SparseMatrixCSC}, Z1::Union{Missing, SparseTensor, SparseMatrixCSC}, Z2::Union{Missing, SparseTensor, SparseMatrixCSC}, Δt::Float64)

An explicit Newmark integrator for 

$$M \ddot{\mathbf{d}} + Z_1 \dot{\mathbf{d}} + Z_2 \mathbf{d} + f = 0$$

The numerical scheme is 

$$\left(\frac{1}{\Delta t^2} M + \frac{1}{2\Delta t}Z_1\right)d^{n+1} = \left(\frac{2}{\Delta t^2} M - \frac{1}{2\Delta t}Z_2\right)d^n - \left(\frac{1}{\Delta t^2} M - \frac{1}{2\Delta t}Z_1\right) d^{n-1} - f$$

To use this integrator, 

```julia 
en = ExplicitNewmark(M, Z1, Z2, Δt)
d2 = step(en, d0, d1, f)
```
"""
struct ExplicitNewmark 
    A::Union{PyObject, Tuple{SparseTensor, PyObject}}
    B::Union{PyObject, SparseTensor} 
    C::Union{PyObject, SparseTensor} 
    function ExplicitNewmark(M::Union{SparseTensor, PyObject, Array{Float64, 2}, SparseMatrixCSC}, 
            Z1::Union{Missing, PyObject, Array{Float64, 2}, SparseTensor, SparseMatrixCSC}, 
            Z2::Union{Missing, PyObject, Array{Float64, 2}, SparseTensor, SparseMatrixCSC}, Δt::Float64)
        M = constant(M)
        if ismissing(Z1)
            A = 1/Δt^2 * M
            C = -1/Δt^2 * M 
        else
            Z1 = constant(Z1)
            A = (1/Δt^2 * M + 1/(2Δt) * Z1)
            C = -(1/Δt^2 * M - 1/(2Δt) * Z1)
        end
        if isa(A, SparseTensor)
            A = factorize(A)
        end
        if ismissing(Z2)
            B = 2/Δt^2 * M
        else
            Z2 = constant(Z2)
            B = (2/Δt^2 * M - Z2)
        end
        new(A, B, C)
    end
end 

function Base.:show(io::IO, en::ExplicitNewmark)
    print("ExplicitNewmark(DOF=$(size(en.B, 1)))")
end

function Base.:step(en::ExplicitNewmark, d0::Union{Array{Float64, 1}, PyObject}, d1::Union{Array{Float64, 1}, PyObject}, f::Union{Array{Float64, 1}, PyObject})
    d0, d1, f = convert_to_tensor([d0, d1, f], [Float64, Float64, Float64])
    en.A \ (en.B * d1 + en.C * d0 - f)
end

