using DelimitedFiles
for i = 1:10
    sleep(3.0)
    writedlm("input.txt", rand(10)); writedlm("input_ready", 1.0) # pay attention to the order
    while true
        if isfile("output_ready") && isfile("output.txt")
            y = readdlm("output.txt")[:]
            println("===============")
            println(y)
            rm("output_ready"); rm("output.txt")
            break
        end
    end
end