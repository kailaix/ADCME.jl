module Optimizer 

using ..ADCME

using LinearAlgebra
using Libdl
#=
The following code is adapted from Flux.jl: https://github.com/FluxML/Flux.jl
The original license from Flux.jl:


The Flux.jl package is licensed under the MIT "Expat" License:
Copyright (c) 2016-19: Julia Computing, INc., Mike Innes and Contributors
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=# 

const ϵ = 1e-8

# TODO: should use weak refs

"""
Descent(η = 0.001)

Classic gradient descent optimiser with learning rate `η`.
For each parameter `p` and its gradient `δp`, this runs `p -= η*δp`

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
the weights.

# Examples
```julia
opt = Descent()

opt = Descent(0.3)

ps = params(model)

gs = gradient(ps) do
  loss(x, y)
end

Flux.Optimise.update!(opt, ps, gs)
```
"""
mutable struct Descent
  eta::Float64
end

Descent() = Descent(0.001)

function apply!(o::Descent, x, Δ)
  Δ .*= o.eta
end

"""
Momentum(η = 0.01, ρ = 0.9)

Gradient descent optimizer with learning rate `η` and momentum `ρ`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
the weights.
- Momentum (`ρ`): Controls the acceleration of gradient descent in the
prominent direction, in effect dampening oscillations.

# Examples
```julia
opt = Momentum()

opt = Momentum(0.01, 0.99)
```
"""
mutable struct Momentum
  eta::Float64
  rho::Float64
  velocity::Union{Missing, Array}
end

Momentum(η = 0.01, ρ = 0.9) = Momentum(η, ρ, missing)

function apply!(o::Momentum, x, Δ)
  η, ρ = o.eta, o.rho
  v = coalesce(o.velocity, zero(x))
  @. v = ρ * v - η * Δ
  ismissing(o.velocity) && (o.velocity = v)
  Δ = -v
end

"""
Nesterov(η = 0.001, ρ = 0.9)

Gradient descent optimizer with learning rate `η` and Nesterov momentum `ρ`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
the weights.
- Nesterov momentum (`ρ`): Controls the acceleration of gradient descent in the
prominent direction, in effect dampening oscillations.

# Examples
```julia
opt = Nesterov()

opt = Nesterov(0.003, 0.95)
```
"""
mutable struct Nesterov
  eta::Float64
  rho::Float64
  velocity::Union{Array, Missing}
end

Nesterov(η = 0.001, ρ = 0.9) = Nesterov(η, ρ, missing)

function apply!(o::Nesterov, x, Δ)
  η, ρ = o.eta, o.rho
  v = coalesce(o.velocity, zero(x))
  d = @. ρ^2 * v - (1+ρ) * η * Δ
  @. v = ρ*v - η*Δ
  ismissing(o.velocity) && (o.velocity = v)
  Δ = -d
end

"""
RMSProp(η = 0.001, ρ = 0.9)

Optimizer using the
[RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
algorithm. Often a good choice for recurrent networks. Parameters other than learning rate
generally don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
the weights.
- Momentum (`ρ`): Controls the acceleration of gradient descent in the
prominent direction, in effect dampening oscillations.

# Examples
```julia
opt = RMSProp()

opt = RMSProp(0.002, 0.95)
```
"""
mutable struct RMSProp
  eta::Float64
  rho::Float64
  acc::Union{Missing, Array}
end

RMSProp(η = 0.001, ρ = 0.9) = RMSProp(η, ρ, missing)

function apply!(o::RMSProp, x, Δ)
  η, ρ = o.eta, o.rho
  acc = coalesce(o.acc, zero(x))
  @. acc = ρ * acc + (1 - ρ) * Δ^2
  ismissing(o.acc) && (o.acc = acc)
  Δ *= η / (√acc + ϵ)
end

"""
ADAM(η = 0.001, β::Tuple = (0.9, 0.999))

[ADAM](https://arxiv.org/abs/1412.6980v8) optimiser.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
second (β2) momentum estimate.

# Examples
```julia
opt = ADAM()

opt = ADAM(0.001, (0.9, 0.8))
```
"""
mutable struct ADAM
  eta::Float64
  beta::Tuple{Float64,Float64}
  state::Union{Missing, Tuple}
end

ADAM(η = 0.001, β = (0.9, 0.999)) = ADAM(η, β, missing)

function apply!(o::ADAM, x, Δ)
  η, β = o.eta, o.beta
  mt, vt, βp = coalesce(o.state, (zero(x), zero(x), β))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
  o.state = (mt, vt, βp .* β)
  return Δ
end

"""
RADAM(η = 0.001, β::Tuple = (0.9, 0.999))

[Rectified ADAM](https://arxiv.org/pdf/1908.03265v1.pdf) optimizer.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
second (β2) momentum estimate.

# Examples
```julia
opt = RADAM()

opt = RADAM(0.001, (0.9, 0.8))
```
"""
mutable struct RADAM
  eta::Float64
  beta::Tuple{Float64,Float64}
  state::Union{Tuple, Missing}
end

RADAM(η = 0.001, β = (0.9, 0.999)) = RADAM(η, β, missing)

function apply!(o::RADAM, x, Δ)
  η, β = o.eta, o.beta
  ρ∞ = 2/(1-β[2])-1
  mt, vt, βp, t = coalesce(o.state, (zero(x), zero(x), β, 1))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  ρ = ρ∞ - 2t*βp[2]/(1-βp[2])
  if ρ > 4
    r = sqrt((ρ-4)*(ρ-2)*ρ∞/((ρ∞-4)*(ρ∞-2)*ρ))
    Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η * r
  else
    Δ =  mt / (1 - βp[1]) * η
  end
  o.state = (mt, vt, βp .* β, t+1)
  return Δ
end

"""
AdaMax(η = 0.001, β::Tuple = (0.9, 0.999))

[AdaMax](https://arxiv.org/abs/1412.6980v9) is a variant of ADAM based on the ∞-norm.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
second (β2) momentum estimate.

# Examples
```julia
opt = AdaMax()

opt = AdaMax(0.001, (0.9, 0.995))
```
"""
mutable struct AdaMax
  eta::Float64
  beta::Tuple{Float64,Float64}
  state::Union{Tuple, Missing}
end

AdaMax(η = 0.001, β = (0.9, 0.999)) = AdaMax(η, β, missing)

function apply!(o::AdaMax, x, Δ)
  η, β = o.eta, o.beta
  mt, ut, βp = coalesce(o.state,  (zero(x), zero(x), β))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. ut = max(β[2] * ut, abs(Δ))
  Δ = (η/(1 - βp[1])) * mt/(ut + ϵ)
  o.state = (mt, ut, βp .* β)
  return Δ
end

"""
ADAGrad(η = 0.1)

[ADAGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) optimizer. It has
parameter specific learning rates based on how frequently it is updated.
Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
the weights.

# Examples
```julia
opt = ADAGrad()

opt = ADAGrad(0.001)
```
"""
mutable struct ADAGrad
  eta::Float64
  acc::Union{Array, Missing}
end

ADAGrad(η = 0.001) = ADAGrad(η, missing)

function apply!(o::ADAGrad, x, Δ)
  η = o.eta
  acc = coalesce(o.acc, fill!(zero(x), ϵ))
  @. acc += Δ^2
  ismissing(o.acc) && (o.acc = acc)
  Δ *= η / (√acc + ϵ)
end

"""
ADADelta(ρ = 0.9)

[ADADelta](https://arxiv.org/abs/1212.5701) is a version of ADAGrad adapting its learning
rate based on a window of past gradient updates.
Parameters don't need tuning.

# Parameters
- Rho (`ρ`): Factor by which the gradient is decayed at each time step.

# Examples
```julia
opt = ADADelta()

opt = ADADelta(0.89)
```
"""
mutable struct ADADelta
  rho::Float64
  state::Union{Tuple, Missing}
end

ADADelta(ρ = 0.9) = ADADelta(ρ, missing)

function apply!(o::ADADelta, x, Δ)
  ρ = o.rho
  acc, Δacc = coalesce(o.state, (zero(x), zero(x)))
  @. acc = ρ * acc + (1 - ρ) * Δ^2
  Δ *= √Δacc/ (√acc + ϵ)
  Δacc = ρ * Δacc + (1 - ρ) * Δ^2
  if ismissing(o.state) 
    o.state = (acc, Δacc)
  end
  return Δ
end

"""
AMSGrad(η = 0.001, β::Tuple = (0.9, 0.999))

The [AMSGrad](https://openreview.net/forum?id=ryQu7f-RZ) version of the ADAM
optimiser. Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
second (β2) momentum estimate.

# Examples
```julia
opt = AMSGrad()

opt = AMSGrad(0.001, (0.89, 0.995))
```
"""
mutable struct AMSGrad
  eta::Float64
  beta::Tuple{Float64, Float64}
  state::Union{Tuple, Missing}
end

AMSGrad(η = 0.001, β = (0.9, 0.999)) = AMSGrad(η, β, missing)

function apply!(o::AMSGrad, x, Δ)
  η, β = o.eta, o.beta
  mt, vt, v̂t = coalesce(o.state, (fill!(zero(x), ϵ), fill!(zero(x), ϵ), fill!(zero(x), ϵ)))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ ^ 2
  @. v̂t = max(v̂t, vt)
  if ismissing(o.state)
    o.state = (mt, vt, v̂t)
  end
  Δ = η * mt / (√v̂t + ϵ)
end

"""
NADAM(η = 0.001, β::Tuple = (0.9, 0.999))

[NADAM](http://cs229.stanford.edu/proj2015/054_report.pdf) is a Nesterov variant of ADAM.
Parameters don't need tuning.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
second (β2) momentum estimate.

# Examples
```julia
opt = NADAM()

opt = NADAM(0.002, (0.89, 0.995))
```
"""
mutable struct NADAM
  eta::Float64
  beta::Tuple{Float64, Float64}
  state::Union{Tuple, Missing}
end

NADAM(η = 0.001, β = (0.9, 0.999)) = NADAM(η, β, missing)

function apply!(o::NADAM, x, Δ)
  η, β = o.eta, o.beta
  mt, vt, (β1p, β2p) = coalesce(o.state, (zero(x), zero(x), o.beta))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  Δ = (β[1] * mt / (1 - β[1] * β1p) + (1 - β[1]) * Δ / (1 - β1p)) / (√(vt * β[2] / (1 - β2p)) + ϵ) * η
  o.state = (mt, vt, (β1p * β[1], β2p * β[2]))
  return Δ
end


"""
    InvDecay(γ = 0.001)
Apply inverse time decay to an optimiser, so that the effective step size at
iteration `n` is `eta / (1 + γ * n)` where `eta` is the initial step size.
The wrapped optimiser's step size is not modified.
# Examples
```julia
decay = InvDecay()
d = apply!(decay, x0, g)
```
"""
mutable struct InvDecay
  gamma::Float64
  state::Union{Int64, Missing}
end

InvDecay(γ = 0.001) = InvDecay(γ, missing)

function apply!(o::InvDecay, x, Δ)
  γ = o.gamma
  n = coalesce(o.state, 1)
  Δ *= 1 / (1 + γ * n)
  o.state = n + 1
  return Δ
end

"""
    ExpDecay(η = 0.001, decay = 0.1, decay_step = 1000, clip = 1e-4)
Discount the learning rate `η` by the factor `decay` every `decay_step` steps till
a minimum of `clip`.
# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- `decay`: Factor by which the learning rate is discounted.
- `decay_step`: Schedule decay operations by setting the number of steps between
                two decay operations.
- `clip`: Minimum value of learning rate.
# Examples
To apply exponential decay to an optimiser:
```julia
decay = ExpDecay()
d = apply!(decay, x0, g)
```
"""
mutable struct ExpDecay
  eta::Float64
  decay::Float64
  step::Int64
  clip::Float64
  current::Union{Int64, Missing}
end

ExpDecay(opt = 0.001, decay = 0.1, decay_step = 1000, clip = 1e-4) = ExpDecay(opt, decay, decay_step, clip, missing)

function apply!(o::ExpDecay, x, Δ)
  η, s, decay = o.eta, o.step, o.decay
  n = o.current = coalesce(o.current, 0) + 1
  if o.current%s == 0 && o.current÷s == 1
    η = max(η * decay, o.clip)
    o.eta = η
  end
  Δ *= η
  return Δ
end

#=
The following code is adapted from Optim.jl: https://github.com/JuliaNLSolvers/Optim.jl
Original license:

Optim.jl is licensed under the MIT License:
Copyright (c) 2012: John Myles White, Tim Holy, and other contributors. Copyright (c) 2016: Patrick Kofod Mogensen, John Myles White, Tim Holy, and other contributors. Copyright (c) 2017: Patrick Kofod Mogensen, Asbjørn Nilsen Riseth, John Myles White, Tim Holy, and other contributors.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
© 2020 GitHub, Inc.
=#

function twoloop!(s,
  gr,
  rho,
  dx_history,
  dg_history,
  m::Integer,
  pseudo_iteration::Integer,
  alpha,
  q,
  scaleinvH0::Bool,
  precon)
  # Count number of parameters
  n = length(s)
  
  # Determine lower and upper bounds for loops
  lower = pseudo_iteration - m
  upper = pseudo_iteration - 1
  
  # Copy gr into q for backward pass
  copyto!(q, gr)
  
  # Backward pass
  for index in upper:-1:lower
    if index < 1
      continue
    end
    i   = mod1(index, m)
    dgi = dg_history[i]
    dxi = dx_history[i]
    @inbounds alpha[i] = rho[i] * real(dot(dxi, q))
    @inbounds q .-= alpha[i] .* dgi
  end
  
  # Copy q into s for forward pass
  # apply preconditioner if precon != nothing
  # (Note: preconditioner update was done outside of this function)
  if scaleinvH0 == true && pseudo_iteration > 1
    # Use the initial scaling guess from
    # Nocedal & Wright (2nd ed), Equation (7.20)
    
    #=
    pseudo_iteration > 1 prevents this scaling from happening
    at the first iteration, but also at the first step after
    a reset due to invH being non-positive definite (pseudo_iteration = 1).
    TODO: Maybe we can still use the scaling as long as iteration > 1?
    =#
    i = mod1(upper, m)
    dxi = dx_history[i]
    dgi = dg_history[i]
    scaling = real(dot(dxi, dgi)) / sum(abs2, dgi)
    @. s = scaling*q
  else
    if isnothing(precon)
      copyto!(s, q)
    else
      ldiv!(s, precon, q)
    end
  end
  # Forward pass
  for index in lower:1:upper
    if index < 1
      continue
    end
    i = mod1(index, m)
    dgi = dg_history[i]
    dxi = dx_history[i]
    @inbounds beta = rho[i] * real(dot(dgi, s))
    @inbounds s .+= dxi .* (alpha[i] - beta)
  end
  
  # # Negate search direction
  # rmul!(s, eltype(s)(-1))
  
  return
end

mutable struct LBFGS
  m::Int
  P
  precondprep!::Function 
  scaleinvH0::Bool
  pseudo_iteration::Int64
  rho::Union{Missing, Vector{<:Real}}
  dx_history
  dg_history
  twoloop_alpha
  twoloop_q
  dx::Union{Missing, Array{<:Real}}
  dg::Union{Missing, Array{<:Real}}
  g_previous::Union{Missing, Array{<:Real}}
  x_previous::Union{Missing, Array{<:Real}}
  s::Union{Missing, Array{<:Real}} # current search direction
end


"""
LBFGS(; m::Integer = 10,
P=nothing,
precondprep = (P, x) -> nothing,
scaleinvH0::Bool = true && (typeof(P) <: Nothing) )

Returns a LBFGS optimizer. `P` is the preconditioner for the LBFGS algorithm. When `P` is not nothing,
`precondprep` modifies `P` in place. When `P` is nothing, a scalar matrix is used as the preconditioner. 
"""
function LBFGS(; m::Integer = 10,
  P=nothing,
  precondprep = (P, x) -> nothing,
  scaleinvH0::Bool = true && (typeof(P) <: Nothing) )
  LBFGS(Int(m), P, precondprep, scaleinvH0, 0, 
  missing, missing, missing, missing, missing, missing, missing, missing, missing, missing)
end


function apply!(o::LBFGS, x, Δ)
  if !ismissing(o.rho)
    update_h!(o, x, Δ)
  end
  
  # the first time some fields in LBFGS have access to the data 
  if ismissing(o.rho)
    T = eltype(x)
    o.rho = Vector{T}(undef, o.m)
    o.dx_history = [similar(x) for i = 1:o.m]
    o.dg_history = [similar(x) for i = 1:o.m]
    o.twoloop_alpha = Vector{T}(undef, o.m)
    o.twoloop_q = similar(x)
    o.dx = similar(x)
    o.dg = similar(x)
    o.g_previous = similar(x)
    o.s = similar(x)
    o.x_previous = copy(x)
  end
  
  # Increment the number of steps we've had to perform
  o.pseudo_iteration += 1
  
  # update the preconditioner
  o.precondprep!(o.P, x)
  
  # Determine the L-BFGS search direction # FIXME just pass state and method?
  twoloop!(o.s, Δ, o.rho, o.dx_history, o.dg_history,
  o.m, o.pseudo_iteration,
  o.twoloop_alpha, o.twoloop_q, o.scaleinvH0, o.P)
  
  copyto!(o.g_previous, Δ)
  
  return o.s 
end


function update_h!(o::LBFGS, x, d)
  n = length(x)
  # Measure the change in the gradient
  o.dg .= d .- o.g_previous
  o.dx, o.x_previous = x - o.x_previous, copy(x)
  
  # Update the L-BFGS history of positions and gradients
  rho_iteration = one(eltype(o.dx)) / real(dot(o.dx, o.dg))
  if isinf(rho_iteration)
    # TODO: Introduce a formal error? There was a warning here previously
    error(DomainError("rho is Inf"))
  end
  idx = mod1(o.pseudo_iteration, o.m)
  @inbounds o.dx_history[idx] .= o.dx
  @inbounds o.dg_history[idx] .= o.dg
  @inbounds o.rho[idx] = rho_iteration
end


mutable struct AndersonAcceleration
    mem::Int64
    atype::Int64
    initilized::Bool 
end

"""
    AndersonAcceleration(; mem::Int64 = 50, atype::Int64 = 1)

Creates an Anderson accelerator. To apply the Anderson Accelerator, do 

```julia
y = apply!(aa, x, Δ)
```

Here `aa` is an AndersonAcceleration instant, `x` is the current state, and `Δ` is the (modified) gradient.

`x` will be modified in place, but is also returned for convenience. 
"""
function AndersonAcceleration(; mem::Int64 = 50, atype::Int64 = 1)
  if Sys.iswindows()
    if !isfile(joinpath(ADCME.LIBDIR, "liblapack.lib"))
      PWD = pwd()
      download("https://github.com/kailaix/ADCME.jl/releases/download/blas_lapack/aa_win.zip", joinpath(ADCME.LIBDIR, "aa_win.zip"))
      cd(ADCME.LIBDIR)
      run(`cmd /c unzip aa_win.zip`)
      files = readdir("aa_win")
      for f in files 
        mv(joinpath("aa_win", f), f, force=true)
      end
      cd(PWD)
    end
  end
  lib = Sys.iswindows() ? "AA.dll" : "libAA.$dlext"
  pth = joinpath(@__DIR__, "..", "deps", "CustomOps", "AndersonAcceleration", "build", lib)
  if !isfile(pth)
    PWD = pwd()
    cd(joinpath(@__DIR__, "..", "deps", "CustomOps", "AndersonAcceleration"))
    compile()
    cd(PWD)
  end
  AndersonAcceleration(mem, atype, false)
end

function apply!(o::AndersonAcceleration, x, Δ)
  n = length(x)
  x_prev = x
  x -= Δ

  lib = Sys.iswindows() ? "AA.dll" : "libAA.$dlext"
  pth = joinpath(@__DIR__, "..", "deps", "CustomOps", "AndersonAcceleration", "build", lib)
  if !o.initilized
    @eval ccall((:init_aa, $pth), Cvoid, (Cint, Cint, Cint), $n, $o.mem, $o.atype)
  end
  @eval ccall((:apply_aa, $pth), Cvoid, (Ref{Cdouble}, Ref{Cdouble}), $x, $x_prev)
  return x 
end

function Base.:finalizer(f, o::AndersonAcceleration)
  lib = Sys.iswindows() ? "AA.dll" : "libAA.$dlext"
  pth = joinpath(@__DIR__, "..", "deps", "CustomOps", "AndersonAcceleration", "build", lib)
  @eval ccall((:finalize_aa, $pth), Cvoid, ())
end


end 