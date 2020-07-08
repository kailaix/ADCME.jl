export test_poisson, generate_random_test_case, generate_simple_test_case
function test_poisson(m=10, n=20)
    u = ones(m, n)
    f = zeros(m, n)
    h = 1.0
    U = zeros(m+2, n+2)
    U[2:end-1, 2:end-1] = u 
    Ut = zeros(m,n)
    for i = 2:m+1
        for j = 2:n+1
            Ut[i-1, j-1] = (U[i+1,j] + U[i-1,j] + U[i,j+1] + U[i,j-1] - h*h*f[i-1,j-1])/4
        end
    end
    U = Ut
    return U 
end

function generate_simple_test_case(m = 10, n = 20, h = 1.0)
    list = []
    u = ones(blockM*m, blockN*n)
    f = zeros(blockM*m, blockN*n)
    U = zeros(blockM*m+2, blockN*n+2)
    U[2:end-1, 2:end-1] = u 
    Ut = zeros(blockM*m, blockN*n)
    for i = 2:blockM*m+1
        for j = 2:blockN*n+1
            Ut[i-1, j-1] = (U[i+1,j] + U[i-1,j] + U[i,j+1] + U[i,j-1] - h*h*f[i-1,j-1])/4
        end
    end
    for i = 1:blockM
        for j = 1:blockN
            u0 = u[(i-1)*m+1:i*m, (j-1)*n+1:j*n]
            f0 = f[(i-1)*m+1:i*m, (j-1)*n+1:j*n]
            U0 = Ut[(i-1)*m+1:i*m, (j-1)n+1:j*n]
            push!(list, (u0, f0, U0))
        end
    end

    @save "test.jld2" list
    return list
end


function generate_random_test_case(m = 10, n = 20, h = 1.0)
    list = []
    u = rand(blockM*m, blockN*n)
    f = rand(blockM*m, blockN*n)
    U = zeros(blockM*m+2, blockN*n+2)
    U[2:end-1, 2:end-1] = u 
    Ut = zeros(blockM*m, blockN*n)
    for i = 2:blockM*m+1
        for j = 2:blockN*n+1
            Ut[i-1, j-1] = (U[i+1,j] + U[i-1,j] + U[i,j+1] + U[i,j-1] - h*h*f[i-1,j-1])/4
        end
    end
    for i = 1:blockM
        for j = 1:blockN
            u0 = u[(i-1)*m+1:i*m, (j-1)*n+1:j*n]
            f0 = f[(i-1)*m+1:i*m, (j-1)*n+1:j*n]
            U0 = Ut[(i-1)*m+1:i*m, (j-1)n+1:j*n]
            push!(list, (u0, f0, U0))
        end
    end

    @save "test.jld2" list
    return list
end
