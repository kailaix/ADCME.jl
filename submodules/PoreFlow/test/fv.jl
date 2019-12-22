using Revise
using ADCME
using PyPlot
using Pore
using SparseArrays
matplotlib.use("agg")

Δt = 0.2
φ = 0.2
c = 0.1

function ρ(p)
    ρ0 = 1
    p_ref = 1
    return @. ρ0*exp(c*(p -p_ref))
end

function dρ(p)
    ρ0 = 1
    p_ref = 1
    return @. ρ0*c*exp(c*(p -p_ref))
end

function Base.:div(x)
    return g.cell_faces * x
end

function avg(x)
    return 0.5 * abs.(g.cell_faces') * x
end

g = RectangleMesh(11,11,0.,0.,11.,11.)
plot(g); xlim(0,1); ylim(0,1); 
savefig("grid.png")

flux = -tpfa(g)



function f(p1, p0)
    u = flux * p1 
    src = zeros(g.num_cells)
    src[101] = 1
    time = φ * (ρ(p1) - ρ(p0))/Δt .* g.cell_volumes
    advection = div(avg(ρ(p1)).*u)
    lhs = time + advection
    rhs = src .* g.cell_volumes
    return lhs - rhs
end

function spdiag0(v::Array{Float64})
    sparse(1:length(v), 1:length(v), v, length(v), length(v))
end

function jac(p, p0)
    u = flux * p 
    Jρ = spdiag0(dρ.(p))
    Ju = flux 
    Javgρ = 0.5 * abs.(g.cell_faces') * Jρ
    φ/Δt * Jρ * spdiag0(g.cell_volumes) + g.cell_faces * ( spdiag0(avg(ρ.(p))) * Ju + spdiag0(u) * Javgρ )
end

p0 = zeros(g.num_cells)
newton_tol = 1e-12
t = .0
T = 1
k = 0
times = [t]

while t<T
    global t, k, p0
    t += Δt 
    k += 1
    push!(times, t)
    err = 1.0
    p = p0
    @info "Solving step $k"
    while err>newton_tol
        @show err
        eq = f(p, p0) 
        p -=  jac(p, p0)\eq
        err = norm(eq)
    end
    p0 = p
    close("all")
    plot(g;cell_value = p, cmap=(0.,1.))
    xlim(0,11)
    ylim(0,11)
    title("time = $k")
    savefig("time$k.png")
end
    
