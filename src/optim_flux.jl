#=
This file is adapted from Flux.jl: https://github.com/FluxML/Flux.jl
The original license from Flux.jl:


The Flux.jl package is licensed under the MIT "Expat" License:

Copyright (c) 2016-19: Julia Computing, INc., Mike Innes and Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=# 

module Optimizer 


# export apply!

const ϵ = 1e-8

# TODO: should use weak refs

"""
    Descent(η = 0.1)

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

Descent() = Descent(0.1)

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
  velocity::IdDict
end

Momentum(η = 0.01, ρ = 0.9) = Momentum(η, ρ, IdDict())

function apply!(o::Momentum, x, Δ)
  η, ρ = o.eta, o.rho
  v = get!(o.velocity, x, zero(x))::typeof(x)
  @. v = ρ * v - η * Δ
  @. Δ = -v
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
  velocity::IdDict
end

Nesterov(η = 0.001, ρ = 0.9) = Nesterov(η, ρ, IdDict())

function apply!(o::Nesterov, x, Δ)
  η, ρ = o.eta, o.rho
  v = get!(o.velocity, x, zero(x))::typeof(x)
  d = @. ρ^2 * v - (1+ρ) * η * Δ
  @. v = ρ*v - η*Δ
  @. Δ = -d
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
  acc::IdDict
end

RMSProp(η = 0.001, ρ = 0.9) = RMSProp(η, ρ, IdDict())

function apply!(o::RMSProp, x, Δ)
  η, ρ = o.eta, o.rho
  acc = get!(o.acc, x, zero(x))::typeof(x)
  @. acc = ρ * acc + (1 - ρ) * Δ^2
  @. Δ *= η / (√acc + ϵ)
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
  state::IdDict
end

ADAM(η = 0.001, β = (0.9, 0.999)) = ADAM(η, β, IdDict())

function apply!(o::ADAM, x, Δ)
  η, β = o.eta, o.beta
  mt, vt, βp = get!(o.state, x, (zero(x), zero(x), β))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
  o.state[x] = (mt, vt, βp .* β)
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
  state::IdDict
end

RADAM(η = 0.001, β = (0.9, 0.999)) = RADAM(η, β, IdDict())

function apply!(o::RADAM, x, Δ)
  η, β = o.eta, o.beta
  ρ∞ = 2/(1-β[2])-1
  mt, vt, βp, t = get!(o.state, x, (zero(x), zero(x), β, 1))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  ρ = ρ∞ - 2t*βp[2]/(1-βp[2])
  if ρ > 4
    r = sqrt((ρ-4)*(ρ-2)*ρ∞/((ρ∞-4)*(ρ∞-2)*ρ))
    @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η * r
  else
    @. Δ =  mt / (1 - βp[1]) * η
  end
  o.state[x] = (mt, vt, βp .* β, t+1)
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
  state::IdDict
end

AdaMax(η = 0.001, β = (0.9, 0.999)) = AdaMax(η, β, IdDict())

function apply!(o::AdaMax, x, Δ)
  η, β = o.eta, o.beta
  mt, ut, βp = get!(o.state, x, (zero(x), zero(x), β))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. ut = max(β[2] * ut, abs(Δ))
  @. Δ = (η/(1 - βp[1])) * mt/(ut + ϵ)
  o.state[x] = (mt, ut, βp .* β)
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
  acc::IdDict
end

ADAGrad(η = 0.1) = ADAGrad(η, IdDict())

function apply!(o::ADAGrad, x, Δ)
  η = o.eta
  acc = get!(o.acc, x, fill!(zero(x), ϵ))::typeof(x)
  @. acc += Δ^2
  @. Δ *= η / (√acc + ϵ)
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
  state::IdDict
end

ADADelta(ρ = 0.9) = ADADelta(ρ, IdDict())

function apply!(o::ADADelta, x, Δ)
  ρ = o.rho
  acc, Δacc = get!(o.state, x, (zero(x), zero(x)))
  @. acc = ρ * acc + (1 - ρ) * Δ^2
  @. Δ *= √Δacc/ (√acc + ϵ)
  @. Δacc = ρ * Δacc + (1 - ρ) * Δ^2
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
  state::IdDict
end

AMSGrad(η = 0.001, β = (0.9, 0.999)) = AMSGrad(η, β, IdDict())

function apply!(o::AMSGrad, x, Δ)
  η, β = o.eta, o.beta
  mt, vt, v̂t = get!(o.state, x, (fill!(zero(x), ϵ), fill!(zero(x), ϵ), fill!(zero(x), ϵ)))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ ^ 2
  @. v̂t = max(v̂t, vt)
  @. Δ = η * mt / (√v̂t + ϵ)
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
  state::IdDict
end

NADAM(η = 0.001, β = (0.9, 0.999)) = NADAM(η, β, IdDict())

function apply!(o::NADAM, x, Δ)
  η, β = o.eta, o.beta
  mt, vt, (β1p, β2p) = get!(o.state, x, (zero(x), zero(x), o.beta))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  @. Δ = (β[1] * mt / (1 - β[1] * β1p) + (1 - β[1]) * Δ / (1 - β1p)) / (√(vt * β[2] / (1 - β2p)) + ϵ) * η
  o.state[x] = (mt, vt, (β1p * β[1], β2p * β[2]))
  return Δ
end


end 