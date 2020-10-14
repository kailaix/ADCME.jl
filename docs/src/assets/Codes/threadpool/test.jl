using ADCME

function test_thread_pool(a)
    test_thread_pool_ = load_op_and_grad("./build/libTestThreadPool","test_thread_pool")
    a = convert_to_tensor(Any[a], [Float64]); a = a[1]
    test_thread_pool_(a)
end

# TODO: specify your input parameters
a = 1.0
u = test_thread_pool(a)
config = tf.ConfigProto(inter_op_parallelism_threads=2, intra_op_parallelism_threads=3, device_count=Dict("CPU"=>1))
sess = Session(config = config); init(sess)
@show run(sess, u)
