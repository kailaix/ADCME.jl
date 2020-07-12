using JLD2
include("poisson.jl")


M = 2
N = 2
m = 500
n = 500
h = 1/(m*M+1)

U = rand(n*N, m*M)
F = zeros(n*N, m*M)

for i = 1:n*N 
    for j = 1:m*M 
        x = j*h
        y = i*h 
        F[i, j] = pi^2 * ( x^2 + y^2 ) * sin( pi * x * y )
    end
end
up = down = zeros(m*M)
left = right = zeros(n*N)
U = zeros(n*N, m*M)

for i = 1:1000
    Unew = poisson_jl(U, up, down, left, right, F, h)
    @info i, norm(U-Unew)
    global U = Unew
end


S = poisson_jl(U, up, down, left, right, F, h)

Us = Array{Array{Float64}}(undef, 2, 2)
Fs = Array{Array{Float64}}(undef, 2, 2)
Ss = Array{Array{Float64}}(undef, 2, 2)
for k1 = 1:2
    for k2 = 1:2
        Us[k1, k2] = U[(k1-1)*n+1:k1*n, (k2-1)*m+1:k2*m]
        Fs[k1, k2] = F[(k1-1)*n+1:k1*n, (k2-1)*m+1:k2*m]
        Ss[k1, k2] = S[(k1-1)*n+1:k1*n, (k2-1)*m+1:k2*m]
    end
end 

@save "data.jld2" Us Fs Ss 