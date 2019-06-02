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