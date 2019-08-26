__precompile__(true)
module JuliaOpModule
    using ADCME
    using PyCall
    global do_it_op 
    function __init__()
        global do_it_op = load_op("$(@__DIR__)/DoItOp/build/libDoItOp", "do_it_op")
    end

    export DoIt!, do_it
    function DoIt!(x::Array{Float64}, y::Array{Float64})
        x[:] = y
    end
    function do_it(o::PyObject)
        do_it_op(o)
    end
end