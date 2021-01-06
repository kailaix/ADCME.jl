@testset "pcl_square_sum" begin 
    using Random; Random.seed!(233)
    y = placeholder(rand(10))
    loss = sum((y-rand(10))^2)
    g = tf.convert_to_tensor(gradients(loss, y))
    function test_f(x0)
        run(sess, g, y=>x0), pcl_square_sum(length(y))
    end
    err1, err2 = test_jacobian(test_f, rand(10); showfig=false)
    @test all(err1.>err2)
end

@testset "pl_hessian" begin 
    using Random; Random.seed!(233)
    x = placeholder(rand(10))
    z = (x[1:5]^2 + x[6:end]^3) * sum(x)
    y = [reshape(sum(z[1:3]), (-1,));z]
    loss = sum((y-rand(6))^2)
    H, dW = pcl_hessian(y, x, loss)

    g = gradients(loss, x)
    function test_f(x0)
        y0, H0 = run(sess, [g, H], feed_dict = Dict(
            x=>x0, 
            dW => pcl_square_sum(length(y))
        ))
    end
    err1, err2 = test_hessian(test_f, rand(length(x)); showfig = false)
    @test all(err1.>err2)
end

@testset "pcl_linear_op" begin 
    using Random; Random.seed!(233)
    x = placeholder(rand(10))
    A = rand(6, 10)
    y = A * x + rand(6)
    loss = sum((y-rand(6))^2)
    J = jacobian(y, x)
    g = gradients(loss, x)
    function test_f(x0)
        gval, Jval = run(sess, [g, J], x=>x0)
        H = pcl_linear_op(Array(Jval'), pcl_square_sum(6))
        return gval, H 
    end
    err1, err2 = test_hessian(test_f, rand(length(x)); showfig = false)
    @test all(err1.>err2)
end


@testset "pcl_sparse_solve" begin 
    using Random; Random.seed!(233)
    A = sprand(10,10,0.6)
    II, JJ, VV = findnz(A)
    x = placeholder(rand(length(VV)))
    temp = SparseTensor(II,JJ,x,10,10)
    indices = run(sess, temp.o.indices)
    B = RawSparseTensor(constant(indices), x, 10, 10)

    rhs = rand(10)
    sol = B\rhs
    loss = sum(sol^2)
    g = gradients(loss, sol)
    G = gradients(loss, x)

    indices = run(sess, B.o.indices) .+ 1
    function test_f(x0)
        s, gval, Gval = run(sess, [sol, g, G], x=>x0)
        H = pcl_sparse_solve(indices, x0, 
            s, 
            pcl_square_sum(length(sol)), gval)
        return Gval, H 
    end
    err1, err2 = test_hessian(test_f, rand(length(x)); showfig = false)
    @test all(err1.>err2)   
end

@testset "pcl_compress" begin 
    using Random; Random.seed!(233)
    if !Sys.iswindows()
        indices = [
            1 1
            1 1
            1 1
            2 2
            3 3
        ]
        pl = placeholder(rand(5))
        A = RawSparseTensor(constant(indices)-1, pl, 3, 3)
        B = compress(A)
        loss = sum(B.o.values^2)
        g = gradients(loss, pl)
        function test_f(x0)
            G = run(sess, g, pl=>x0)
            W = pcl_square_sum(3)
            J = pcl_compress(indices)
            H = pcl_linear_op(J, W)
            G, H 
        end

        x0 = rand(length(pl))
        err1, err2 = test_hessian(test_f, x0; showfig = false)
        @test all(err1.>err2)   
    end
end