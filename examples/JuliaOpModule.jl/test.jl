using ADCME
do_it = load_op("./build/libDoItOp", "do_it_op")

function myfun(y, x)
    @. y = 2x
end

x = constant(rand(100))
y = 2x # or `y = Variable(rand(100))`
u = do_it(y)
config = tf.ConfigProto(inter_op_parallelism_threads=1)
sess = tf.Session()
init(sess)
ccall((:get_id, "./build/libDoItOp.so"), Cvoid, ())
run(sess, u, x=>rand(100))
