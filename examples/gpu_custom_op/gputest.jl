using ADCME

function gpu_test(a)
    gpu_test_ = load_op_and_grad("$(@__DIR__)/build/libGpuTest","gpu_test")
    a = convert_to_tensor([a], [Float64]); a = a[1]
    gpu_test_(a)
end

# TODO: specify your input parameters
a = rand(3)
u = gpu_test(a)
sess = Session(); init(sess)
v1 = run(sess, u)
v2 = 2a
println("Computed: $v1; expected: $v2")
 
