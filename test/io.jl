@testset "save and load" begin
    a = Variable(1.0)
    b = Variable(2.0)
    init(sess)
    save(sess, "temp.mat", [a,b])
    op1 = assign(a, 3.0)
    op2 = assign(b, 3.0)
    run(sess, op1)
    run(sess, op2)

    sess1 = Session()
    load(sess1, "temp.mat", [a,b])
    @test run(sess1, [a,b])≈[1.0;2.0]
    rm("temp.mat")
end