# Resource Manager

Sometimes we want to store data for different operations to share, or maintain a stateful kernel (data are shared across different invocations). One way to achieve this goal in the concurrency environment is to use  `ResourceMgr` in C++ custom operators. 

A typical usage of `ResourceMgr` is as follows
1. Define your own resource, which should inherent from `ResourceBase` and `DebugString` must be defined (it is an abstract method in `ResourceBase`).
```c++
#include "tensorflow/core/framework/resource_mgr.h"
struct MyVar: public ResourceBase{
  string DebugString() const { return "MyVar"; };
  mutex mu;
  int32 val;
};
```

2. Access the system `ResourceMgr` through 
```c++
auto rm = context->resource_manager();
```

3. Define your resource creation and manipulation method (make sure at any time there is only one single instance given the same container name and resource name).
```c++
MyVar* my_var;
Status s = rm->LookupOrCreate<MyVar>("my_container", "my_name", &my_var, [&](MyVar** ret){
    printf("Create a new container\n");
    *ret = new MyVar;
    (*ret)->val = *u_tensor;
    return Status::OK();
});
DCHECK_EQ(s, Status::OK());
my_var->val += 1;
my_var->Unref();
```

When using the `ResourceMgr`, keep in mind that whenever you execute a new path in the computational graph, the system will create a new `ResourceMgr`. Therefore, to run operators that manipulate `ResourceMgr` in parallel, the trigger operator (which is fed to `run(sess, ...)`) must be attached those manipulation dependencies. 

See the following scripts for an example

[CMakeLists.txt](https://kailaix.github.io/ADCME.jl/dev/assets/Codes/ResourceManager/CMakeLists.txt), [TestResourceManager.cpp](https://kailaix.github.io/ADCME.jl/dev/assets/Codes/ResourceManager/TestResourceManager.cpp), [gradtest.jl](https://kailaix.github.io/ADCME.jl/dev/assets/Codes/ResourceManager/gradtest.jl)

