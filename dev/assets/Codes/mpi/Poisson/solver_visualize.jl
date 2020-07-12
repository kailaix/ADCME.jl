
include("poisson.jl")

results = []
@load  "solution00.jld2" result
push!(results, result[:,:])
@load  "solution01.jld2" result
push!(results, result[:,:]) 
@load  "solution10.jld2" result
push!(results, result[:,:]) 
@load  "solution11.jld2" result
push!(results, result[:,:])
M = 2
N = 2
n, m = size(result)

R = zeros(n*N, m*M)
for k1 = 1:2
    for k2 = 1:2
        R[(k1-1)*n+1:k1*n, (k2-1)*m+1:k2*m] = results[(k1-1)*2+k2]
    end
end 

pcolormesh(R)
colorbar()