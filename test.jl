begin 
    a = 1
    deps = 2
    @goto deps 

    a += 1
    @label deps 

    a += 1
end