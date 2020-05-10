using ADCME

save_tensor = load_op_and_grad("./build/libSaveTensor","save_tensor")
get_tensor = load_op_and_grad("./build/libGetTensor","get_tensor")
delete_tensor = load_op_and_grad("./build/libDeleteTensor","delete_tensor")

val = constant(rand(10))
t1 = tf.constant("tensor1")
t2 = tf.constant("tensor2")
t3 = tf.constant("tensor3")
u1 = save_tensor(t1,val)
u2 = save_tensor(t2,2*val)
u3 = save_tensor(t3,3*val)

z1 = get_tensor(t1);
z2 = get_tensor(t2);
z3 = get_tensor(t3);

d1 = delete_tensor(t1);
d2 = delete_tensor(t2);
d3 = delete_tensor(t3);
sess = Session(); 
run(sess, [u1,u2,u3])


run(sess, z1)
run(sess, z2)
run(sess, z3)
run(sess, d2)

