# launch the script with
# mpirun -n 10 julia test_poisson.jl 
# We divide the mesh into 2x5 blocks, each block consists of a grid of size 800x800
M = 2
N = 5
m = 3000
n = 3000

mm = m*M 
nn = n*N 

u0 = zeros(mm+2, nn+2) # initial guess
f0 = ones(mm, nn) # right hand size
h = 0.01 # mesh size 
u1 = zeros(mm, nn)
@timed begin 
    for k = 1:20
        @info k
        for ii = 1:mm
            for jj = 1:nn  
                i = ii+1
                j = jj+1
                u1[ii,jj] = (u0[i-1,j] + u0[i,j-1]+u0[i+1,j] + u0[i,j+1] - h*h*f0[ii,jj])/4
            end
        end
        u0[2:end-1, 2:end-1] = u1
    end
end