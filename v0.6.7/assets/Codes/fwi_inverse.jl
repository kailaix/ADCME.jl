using ADCME
using PyPlot 
using ADCMEKit 

n = 200
pml = 30
C = 1000.0
NT = 1000
Δt = 1.0/NT 
x0 = LinRange(0, 1, n+1)
h = 1/n 
xE = Array((0:n+2pml)*h .- pml*h)
xH = (xE[2:end]+xE[1:end-1])/2 
N = n + 2pml + 1

σE = zeros(N)
for i = 1:pml
    d = i*h
    σE[pml + n + 1 + i] = C* (d/(pml*h))^3
    σE[pml+1-i] = C* (d/(pml*h))^3
end

σH = zeros(N-1)
for i = 1:pml 
    d = (i-1/2)*h 
    σH[pml + n + i] = C* (d/(pml*h))^3
    σH[pml+1-i] = C* (d/(pml*h))^3
end

function ricker(dt = 0.002, f0 = 5.0)
    nw = 2/(f0*dt)
    nc = floor(Int, nw/2)
    t = dt*collect(-nc:1:nc)
    b = (π*f0*t).^2
    w = @. (1 - 2b)*exp(-b)
end
R = ricker()
if length(R)<NT+1
    R = [R;zeros(NT+1-length(R))]
end
R = R[1:NT+1]
# tn = ( 0:NT ) *Δt
# R = @. exp( -20(tn-0.3)^2)
# error()
R_ = constant(R)


cH = ones(length(xH)) * 2.
cH[pml + Int(round(0.5÷h)): pml + Int(round(0.75÷h))] .= 1.0
cH[pml + Int(round(0.75÷h)): end] .= 1.

mask = ones(length(xH))
mask[pml+1:end-pml] .= 0.0
c_ = Variable(ones(N-2pml))

x0 = collect(((pml:N-pml -1) .- pml)*h)
# c_ = squeeze(fc(x0, [20,20,1])) + 1.0
# c_ = 2tanh(c_) + 2.5
cH = scatter_update(constant(cH.*mask), pml+1:N-pml, c_)
# cH = constant(cH)
cE = (cH[1:end-1]+cH[2:end])/2
Z = zeros(N)
Z[pml + pml÷2] = 1.0
Z = Z[2:end-1]

function condition(i, E_arr, H_arr)
    i<=NT+1
end

function body(i, E_arr, H_arr)
    E = read(E_arr, i-1)
    H = read(H_arr, i-1)
    ΔH = cH * (E[2:end]-E[1:end-1])/h - σH*H
    H += ΔH * Δt
    ΔE = cE * (H[2:end]-H[1:end-1])/h - σE[2:end-1]*E[2:end-1] + R_[i] * Z
    E = scatter_add(E, 2:N-1, ΔE * Δt)
    # E = scatter_update(E, N÷2, R_[i])
    i+1, write(E_arr, i, E), write(H_arr, i, H)
end

E_arr = TensorArray(NT+1)
H_arr = TensorArray(NT+1)

E_arr = write(E_arr, 1, zeros(N))
H_arr = write(H_arr, 1, zeros(N-1))

i = constant(2, dtype = Int32)

_, E, H = while_loop(condition, body, [i, E_arr, H_arr])

E = stack(E); E = set_shape(E, (NT+1, N))
H = stack(H)

loss = sum((E[:,pml+pml÷2] - E_[:, pml+pml÷2])^2)

sess = Session(); init(sess)
BFGS!(sess, loss)