reset_default_graph(); sess = Session()
# not testable
# # this must be the test
# @testset "Eager evaluation" begin   
#     reset_default_graph()
#     sess = Session()
#     enable_eager_execution()
#     g = rand(10,10)
#     o = Variable(g)
#     @test value(o) ≈ g
# end

@testset "control_dependency" begin   
    a = Variable(ones(10))
    b = Variable(ones(10))
    for i = 1:10
        control_dependencies(a) do
            a = scatter_add(a, i, b[i])
        end
    end

    a_ = constant(0.0)
    b_ = Variable(0.0)
    op = assign(b_, 2.0)
    a_ = a_ + 1.0
    a_ = bind(a_, op)


    init(sess)
    run(sess, a_)
    @test run(sess, a)≈ones(10)*2
    @test run(sess, b_)≈2.0


    a = Variable(ones(10))
    b = Variable(ones(10))
    for i = 1:10
        control_dependencies(a) do
            a = scatter_add(a, i, b[i])
        end
    end

    a_ = spdiag(ones(10))
    b_ = Variable(0.0)
    op = assign(b_, 2.0)
    a_ = a_*2
    a_ = bind(a_, op)


    init(sess)
    run(sess, a_)
    @test run(sess, a)≈ones(10)*2
    @test run(sess, b_)≈2.0


end

@testset "while loop" begin
    a = constant(rand(10))
    condition = (i, var)->tf.less(i,11)
    function body(i, var)        
        var = tf.cond(tf.equal(i, 1), 
            ()->write(var, i, a),
            ()->write(var, i, 2*read(var, i-1))
        )
        i+1, var
    end
    i = constant(1, dtype=Int32) 
    ta = TensorArray(10)
    out_i, out_ta = tf.while_loop(condition, body, [i,ta])
    ts = read(out_ta, 10)
    sum_ts = sum(ts)
    grd = gradients(sum_ts, a)
    @test run(sess, grd)≈512*ones(10)


    ta = TensorArray(10)
    a = constant(1.0)
    ta = write(ta, 1, a)
    ta = write(ta, 2, read(ta,1)+a) # 2a
    ta = write(ta, 3, read(ta,2)+a) # 3a
    ta = write(ta, 4, read(ta,2)+read(ta,3)+a) # 6a
    g = gradients(read(ta, 4), a)
    @test run(sess, g)≈6


    ta = TensorArray(9)

    function condition(i, ta)
        tf.less(i, 10)
    end
    ta = write(ta, 1, constant(1.0))
    ta = write(ta, 2, constant(1.0))
    function body(i, ta)
        ta = write(ta, i, 2*read(ta,i-1)+read(ta,i-2)+a)
        i+1, ta
    end

    i = constant(3, dtype=Int32)
    _, ta_out = while_loop(condition, body, [i,ta])
    g = gradients(read(ta_out,9),a)
    @test run(sess, g)≈288

end


@testset "if_clause" begin
    a = constant(2.0)
    b = constant(1.0)
    c = constant(0.0)
    bb = b*2
    cc = c + 3.0
    res = if_else(a>1.0, bb, cc)
    @test run(sess, res)≈2.0
end

@testset "if_else: tf.where" begin
    condition = [true true false false
                true true true true]
    a = [1 1 1 1
         1 1 1 1]
    b = [2 2 2 2
         2 2 2 2]
    res = if_else(condition, a, b)
    @test run(sess, res)≈[1 1 2 2
                          1 1 1 1]

    res1 = if_else(condition[1,:], a[1,:], b[1,:])
    @test run(sess, res1)≈[1;1;2;2]
end

@testset "get and add collection" begin
    a = Variable(ones(10), name="my_collect1")
    b = 2a 
    @test get_collection("my_collect")==[a]
end

@testset "has_gpu" begin
    @test has_gpu() in [true, false]
end

@testset "timeline" begin
    a = normal(2000, 5000)
    b = normal(5000, 1000)
    res = a*b 
    run_profile(sess, res)
    save_profile("test.json")
    rm("test.json")
end

@testset "independent" begin 
    x = constant(rand(10))
    y = 2*x 
    z = sum(independent(y))
    @test isnothing(gradients(z, x))
end

@testset "run corner cases" begin 
    pl = placeholder(0.1)
    @test run(sess, pl, pl=>4.0) ≈ 4.0
end

@testset "@cpu @gpu" begin
    @cpu 2 begin 
        global a = constant(rand(10,10))
    end
    @gpu 2 begin 
        global b = constant(rand(10,10))
    end
    @test a.device == "/device:CPU:2"
    @test b.device == "/device:GPU:2"

    @cpu begin 
        global a = constant(rand(10,10))
    end
    @gpu begin 
        global b = constant(rand(10,10))
    end
    @test a.device == "/device:CPU:0"
    @test b.device == "/device:GPU:0"
end

@testset "mpi utils" begin 
    @test_nowarn has_mpi() 
    try 
        get_mpi()
    catch
        @test true 
    end
    try 
        get_mpirun()
    catch
        @test true 
    end
end