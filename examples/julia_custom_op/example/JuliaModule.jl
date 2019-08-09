module JuliaModule

using ForwardDiff

function relu(x)
    return max(0.0, x)
end

function forward(x, b1, w1, b2, w2)
    res = w2*relu.(w1*x .+ b1) .+ b2
end


Base.@ccallable function forward(x::Array{Float64}, b1::Array{Float64}, w1::Array{Float64}, b2::Array{Float64}, w2::Array{Float64}, y::Array{Float64})::Nothing
    v = (size(x), size(w1), size(b1), size(w2), size(b2))
    println("$v")
    @show x, b1, w1, b2, w2
    println(w2*relu.(w1*x .+ b1) .+ b2)
    res = w2*relu.(w1*x .+ b1) .+ b2
    println(res)
    y[:] = res
    return nothing
end

Base.@ccallable function backward(x::Array{Float64}, b1::Array{Float64}, w1::Array{Float64}, b2::Array{Float64}, w2::Array{Float64}, J::Array{Float64})::Nothing
    f = x->forward(x, b1, w1, b2, w2)
    J_ = ForwardDiff.jacobian(f, x)
    J[:] = J_
    return nothing
end

end