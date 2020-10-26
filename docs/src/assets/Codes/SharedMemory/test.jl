using ADCME

if Sys.iswindows()
    global save_tensor = load_op_and_grad("./build/Release/libSaver","save_tensor")
    global get_tensor = load_op_and_grad("./build/Release/libSaver","get_tensor")
    global delete_tensor = load_op_and_grad("./build/Release/libSaver","delete_tensor")
else 
    global save_tensor = load_op_and_grad("./build/libSaveTensor","save_tensor")
    global get_tensor = load_op_and_grad("./build/libGetTensor","get_tensor")
    global delete_tensor = load_op_and_grad("./build/libDeleteTensor","delete_tensor")
end 

val = constant(rand(10))
t1 = constant("tensor1")
t2 = constant("tensor2")
t3 = constant("tensor3")
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
run(sess, [u1,u2,u3]) # add all the keys

# get the keys one by one
run(sess, z1)
run(sess, z2)
run(sess, z3)

# delete 2nd key
run(sess, d2)