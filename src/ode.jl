export runge_kutta
function runge_kutta_one_step(f::Function, t::PyObject, y::PyObject, Δt::PyObject, θ::Union{PyObject, Missing})
    k1 = Δt*f(t, y, θ)
    k2 = Δt*f(t+Δt/2, y+k1/2, θ)
    k3 = Δt*f(t+Δt/2, y+k2/2, θ)
    k4 = Δt*f(t+Δt, y+k3, θ)
    y = y + k1/6 + k2/3 + k3/3 + k4/6
end

@doc raw"""
    runge_kutta(y::Union{PyObject, Float64, Array{Float64}}, T::Union{PyObject, Float64}, 
                NT::Union{PyObject,Int64}, f::Function, θ::Union{PyObject, Missing}=missing)

Solves 
```math
\frac{dy}{dt} = f(y, t, \theta)
```
with Runge-Kutta method. 

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
function runge_kutta(y::Union{PyObject, Float64, Array{Float64}}, T::Union{PyObject, Float64}, 
                NT::Union{PyObject,Int64}, f::Function, θ::Union{PyObject, Missing}=missing)
    y = convert_to_tensor(y)
    Δt = convert_to_tensor(T/NT)

    ta = TensorArray(NT+1) # storing y
    function condition(i, ta)
        i <= NT+1
    end
    function body(i, ta)
        y = read(ta, i-1)
        y_ = runge_kutta_one_step(f, (cast(eltype(Δt), i)-1)*Δt, y, Δt, θ)
        ta = write(ta, i, y_)
        i+1, ta 
    end
    ta = write(ta, 1, y)
    i = constant(2, dtype=Int32)
    _, out = while_loop(condition, body, [i, ta])
    res = stack(out)
end