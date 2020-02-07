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
    rm("temp.mat", force=true)
end

@testset "psave and pload" begin
py"""
def psave_test_function(x):
    return x+1
"""
    a = py"psave_test_function"
    psave(a, "temp.pkl")
    b = pload("temp.pkl")
    @test b(2)==3
    rm("temp.pkl", force=true)
end

@testset "diary" begin
    d = Diary("test")
    b = constant(1.0)
    a = scalar(b)
    for i = 1:100
        write(d, i, run(sess, a))
    end
    # activate(d)
    save(d, "mydiary")
    e = Diary("test2")
    load(e, "mydiary")
    # activate(e)

    try;rm("mydiary", recursive=true, force=true);catch;end
end