using ForwardDiff
using DelimitedFiles

function relu(x)
    return max(0.0, x)
end

function forward(x, b1, w1, b2, w2)
    res = w2*relu.(w1*x .+ b1) .+ b2
end


function forward(x::Array{Float64}, b1::Array{Float64}, w1::Array{Float64}, b2::Array{Float64}, w2::Array{Float64})
    v = (size(x), size(w1), size(b1), size(w2), size(b2))
    println("$v")
    @show x, b1, w1, b2, w2
    println(w2*relu.(w1*x .+ b1) .+ b2)
    res = w2*relu.(w1*x .+ b1) .+ b2
    println(res)
    return res
end

function backward(x::Array{Float64}, b1::Array{Float64}, w1::Array{Float64}, b2::Array{Float64}, w2::Array{Float64})
    f = x->forward(x, b1, w1, b2, w2)
    J = ForwardDiff.jacobian(f, x)
    return J
end

b1 = ones(10)
b2 = ones(10)
W1 = ones(10,10)
W2 = ones(10,10)
println("Serving...")
while true
    if isfile("input_ready") && isfile("input.txt")
        println("Received file")
        x = readdlm("input.txt")[:]
        y = forward(x, b1, W1, b2, W2)
        writedlm("output.txt", y)
        rm("input.txt"); rm("input_ready") # or mark as done
        writedlm("output_ready", [1.0]) # must be written after output.txt
    end
end

