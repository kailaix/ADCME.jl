using PyPlot
using JLD2
function simulate(σ, κ, τ)
    z = zeros(10000)
    α = 0.5
    Δt = 0.01
    z[1] = τ
    for i = 2:length(z)
        x = z[i-1]
        W = randn()
        z[i] = (
            x + κ * (τ - α*x)*Δt + σ * sqrt(x) * sqrt(Δt) * W  + 1/4*σ^2*Δt*(W^2-1)
        )/(1+(1-α)*κ*Δt)
    end
    z
end

close("all")
z = simulate(0.08, 0.5, 0.06)
@save "data.jld2" z
plot(z)
savefig("test.png")