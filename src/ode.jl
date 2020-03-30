export ode45, rk4, αscheme, αscheme_time

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
\begin{align}
k_1 &= \Delta t f(t_n, y_n, \theta)\\
k_2 &= \Delta t f(t_n+\Delta t/2, y_n + k_1/2, \theta)\\
k_3 &= \Delta t f(t_n+\Delta t/2, y_n + k_2/2, \theta)\\
k_4 &= \Delta t f(t_n+\Delta t, y_n + k_3, \theta)\\
y_{n+1} &= y_n + \frac{k_1}{6} +\frac{k_2}{3} +\frac{k_3}{3} +\frac{k_4}{6}
\end{align}
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
        ρ::Float64 = 1.0)

Generalized α-scheme. 
$$M u_{tt} + C u_{t} + K u = F$$

`Force` must be an array of size `n`×`p`, where `d0`, `v0`, and `a0` have a size `p`
`Δt` is an array (variable time step). 

The generalized α scheme solves the equation by the time stepping
```math
\begin{align}
\bf d_{n+1} &= \bf d_n + h\bf v_n + h^2 \left(\left(\frac{1}{2}-\beta_2 \right)\bf a_n + \beta_2 \bf a_{n+1}  \right)\\
\bf v_{n+1} &= \bf v_n + h((1-\gamma_2)\bf a_n + \gamma_2 \bf a_{n+1})\\
\bf F(t_{n+1-\alpha_{f_2}}) &= M \bf a _{n+1-\alpha_{m_2}} + C \bf v_{n+1-\alpha_{f_2}} + K \bf{d}_{n+1-\alpha_{f_2}}
\end{align}
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

!!! note 
    In the case $u$ has a nonzero essential boundary condition $u_b$, we let $\tilde u=u-u_b$, then 
    $$M \tilde u_{tt} + C \tilde u_t + K u = F - K u_b - C \dot u_b$$
"""
function αscheme(M::Union{SparseTensor, SparseMatrixCSC}, 
                      C::Union{SparseTensor, SparseMatrixCSC}, 
                      K::Union{SparseTensor, SparseMatrixCSC}, 
                      Force::Union{Array{Float64}, PyObject}, 
                      d0::Union{Array{Float64, 1}, PyObject}, 
                      v0::Union{Array{Float64, 1}, PyObject}, 
                      a0::Union{Array{Float64, 1}, PyObject}, 
                      Δt::Array{Float64}; 
                      solve::Union{Missing, Function} = missing,
                      ρ::Float64 = 1.0)
    n = length(Δt)
    αm = (2ρ-1)/(ρ+1)
    αf = ρ/(1+ρ)
    γ = 1/2-αm+αf 
    β = 0.25*(1-αm+αf)^2
    d = length(d0)

    M = isa(M, SparseMatrixCSC) ? constant(M) : M
    C = isa(C, SparseMatrixCSC) ? constant(C) : C
    K = isa(K, SparseMatrixCSC) ? constant(K) : K
    Force, d0, v0, a0, Δt = convert_to_tensor([Force, d0, v0, a0, Δt], [Float64, Float64, Float64, Float64, Float64])

    function equ(dc, vc, ac, dt, Force)
        dn = dc + dt*vc + dt^2/2*(1-2β)*ac 
        vn = vc + dt*((1-γ)*ac)

        df = (1-αf)*dn + αf*dc
        vf = (1-αf)*vn + αf*vc 
        am = αm*ac 

        rhs = Force - (M*am + C*vf + K*df)
        A = (1-αm)*M + (1-αf)*C*dt*γ + (1-αf)*K*β*dt^2

        if !ismissing(solve)
            return solve(A, rhs)
        else 
            return A\rhs
        end
    end

    function condition(i, tas...)
        return i<=n
    end
    function body(i, tas...)
        dc_arr, vc_arr, ac_arr = tas
        dc = read(dc_arr, i)
        vc = read(vc_arr, i)
        ac = read(ac_arr, i)
        y = equ(dc, vc, ac, Δt[i], Force[i])
        dn = dc + Δt[i]*vc + Δt[i]^2/2*((1-2β)*ac+2β*y)
        vn = vc + Δt[i]*((1-γ)*ac+γ*y)
        i+1, write(dc_arr, i+1, dn), write(vc_arr, i+1, vn), write(ac_arr, i+1, y)
    end

    dM = TensorArray(n+1); vM = TensorArray(n+1); aM = TensorArray(n+1)
    dM = write(dM, 1, d0)
    vM = write(vM, 1, v0)
    aM = write(aM, 1, a0)
    i = constant(1, dtype=Int32)
    _, d, v, a = while_loop(condition, body, [i,dM, vM, aM])
    stack(d), stack(v), stack(a)
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
        tf = (1-αf)*(tc+dt) + αf*tc
    end

    tcs = Float64[]
    tc = 0.0
    for i = 1:n
        push!(tcs, equ(tc, Δt[i]))
        tc += Δt[i]
    end
    return tcs 
end