reset_default_graph(); sess = Session()

@testset "test_jacobian" begin 
    function f(x)
        x.^3, 3*diagm(0=>x.^2)
    end
    err1, err2 = test_jacobian(f, rand(10))
    @test err2[end] < err1[end]
end

@testset "lineview" begin 
    @test begin 
        a = placeholder(rand(10))
        loss = sum((a+1)^2)
        close("all")
        lineview(sess, a, loss, -ones(10))
        savefig("lineview.png")
        true 
    end
end

@testset "meshview" begin 
    @test begin 
        a = placeholder(rand(10))
        loss = sum((a+1)^2)
        close("all")
        meshview(sess, a, loss, -ones(10))
        savefig("meshview.png")
        true 
    end
    @test begin 
        a = placeholder(rand(10))
        loss = sum((a+1)^2)
        close("all")
        pcolormeshview(sess, a, loss, -ones(10))
        savefig("pcolormeshview.png")
        true 
    end
end

@testset "gradview" begin 
    @test begin 
        a = placeholder(rand(10))
        loss = sum((a+1.0)^2)
        close("all")
        gradview(sess, a, loss, rand(10))
        true
    end
end

@testset "jacview" begin 
    @test begin 
        u0 = rand(10)
        function verify_jacobian_f(θ, u)
            r = u^3+u - u0
            r, spdiag(3u^2+1.0)
        end
        close("all")
        jacview(sess, verify_jacobian_f, missing, u0)
        savefig("jacview.png")
        true
    end
end

@testset "PCLview" begin 
end

@testset "animate" begin
    @test begin  
        θ = LinRange(0, 2π, 100)
        x = cos.(θ)
        y = sin.(θ)
        pl, = plot([], [], "o-")
        t = title("0")
        xlim(-1.2,1.2)
        ylim(-1.2,1.2)
        function update(i)
            t.set_text("$i")
            pl.set_data([x[1:i] y[1:i]]'|>Array)
        end
        p = animate(update, 1:100)
        saveanim(p, "anim.gif")
        true
    end
end