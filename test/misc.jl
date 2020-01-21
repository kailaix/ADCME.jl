@testset "poisson" begin
    n = 20
    h = 1/n
    node = zeros((n+1)^2, 2)
    node_mid = zeros(2n^2,2)
    elem = zeros(Int32, 2n^2, 3)
    k = 1
    for i = 1:n
        for j = 1:n
            node[i+(j-1)*(n+1), :] = [(i-1)*h;(j-1)*h]
            node[i+1+(j-1)*(n+1), :] = [i*h;(j-1)*h]
            node[i+(j)*(n+1), :] = [(i-1)*h;(j)*h]
            node[i+1+(j)*(n+1), :] = [(i)*h;(j)*h]
            elem[k,:] = [i+(j-1)*(n+1) i+1+(j-1)*(n+1) i+(j)*(n+1)];
            elem[k+1,:] = [i+1+(j)*(n+1) i+1+(j-1)*(n+1) i+(j)*(n+1)];
            node_mid[k, :] = mean(node[elem[k,:], :], dims=1)
            node_mid[k+1, :] = mean(node[elem[k+1,:], :], dims=1)
            k += 2
        end
    end
    bdnode = Int32[]
    for i = 1:n+1
        for j = 1:n+1
            if i==1 || j == 1 || i==n+1 || j==n+1
                push!(bdnode, i+(j-1)*(n+1))
            end
        end
    end
    x, y = node_mid[:,1], node_mid[:,2]
    f = constant(@. 2y*(1-y)+2x*(1-x))
    a = constant(ones(2n^2))
    u = pde_poisson(node,elem,bdnode,f,a)
    uval = run(sess, u)
    x, y = node[:,1], node[:,2]
    uexact = @. x*(1-x)*y*(1-y)
    @test maximum(abs.(uval - uexact))<1e-3
end