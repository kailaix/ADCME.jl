export ode45, rk4

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

