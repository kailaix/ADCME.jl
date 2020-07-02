# Using MPI with ADCME


Many large-scale scientific computing involves parallel computing. Among many parallel computing models, the MPI  is one of the most popular models. In this section, we describe how ADCME can work with MPI for solving inverse modeling. Specifically, we describe how gradients can be back-propagated via MPI function calls.  

!!! info 
    Message Passing Interface (MPI) is an interface for parallel computing based on message passing models. In the message passing model, a master process assigns work to workers by passing them a message that describes the work. The message may be data or meta information (e.g., operations to perform). A consensus was reached around 1992 and the MPI standard was born. MPI is a definition of interface, and the implementations are left to hardware venders. 

## A First Example

We will utilize the [MPI.jl](https://github.com/JuliaParallel/MPI.jl) package, which is a Julia interface to many MPI implementations (MPICH, MSMPI, etc.). Because MPI.jl is essentially a wrapper of MPI C language implementations, we can freely interact shared library with MPI.jl. For example, in [this directory](docs/src/assets/Codes/mpi), we have a C++ code 

```c++
#include <mpi.h>
#include <iostream>

#ifdef _WIN32
#define EXPORTED __declspec(dllexport)
#else
#define EXPORTED
#endif 

extern "C" EXPORTED int printinfo(){
    MPI_Comm comm = MPI_COMM_WORLD;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[1000];
    int name_len;
    MPI_Get_processor_name( processor_name , &name_len);
    printf("Hello world from processor %s, rank %d out of %d processors\n",
        processor_name, world_rank, world_size);
    int result;
    MPI_Reduce( &world_rank , &result , 1 , MPI_INT , MPI_SUM , 0 , comm);
    return result;
}
```

After we compile it into a shared library, we can write Julia codes that mix the C++ kernel and MPI.jl

```julia
using MPI 

MPI.Init()

v = ccall((:printinfo, "./build/debug/mtest.dll"), Cint, ())
print(v)
```

For example, in the shell we type 
```bash
mpiexec.exe -n 4 julia test.jl
```

We have the following output 
```
13296304
13296304
13296304
6
Hello world from processor LAPTOP-92GNG3GT.stanford.edu, rank 1 out of 4 processors
Hello world from processor LAPTOP-92GNG3GT.stanford.edu, rank 2 out of 4 processors
Hello world from processor LAPTOP-92GNG3GT.stanford.edu, rank 0 out of 4 processors
Hello world from processor LAPTOP-92GNG3GT.stanford.edu, rank 3 out of 4 processors
```

The first three numbers are junk because they are on the worker processors. 

## Custom Operators with MPI

We can also make custom operators with MPI. Let us consider computing

$$f(\theta) = \sum_{i=1}^n f_i(\theta)$$

Each $f_i$ is a very expensive function so it makes sense to use MPI to split the jobs on different processors. To simplify the problem, we consider 

$$f(\theta) = f_1(\theta) + f_2(\theta) + f_3(\theta) + f_4(\theta)$$

where $f_i(\theta) = \theta^{i-1}$. 


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mpi.png?raw=true)

We make two custom operators: `basis` 

$$\texttt{basis}(\theta) = 1, \theta, \theta^2, \theta^3, \ldots, \theta^n$$

and `m_sum`

$$\texttt{m_sum}(a_1, a_2, \ldots, a_n) = a_1 + a_2 + \ldots +a_n$$


Here $n$ is the number of processors. 

To implement `basis`, we need to broadcast $\theta$ at processor 0 to $n$ processors (`MPI_Bcast`). For back-propagating the gradients, for each processor, we need to back-propagate the gradient, respectively, and then aggregate the gradients on processor 0 (`MPI_Reduce`). This leads to the following forward and backward implementation:

```c++
#include "mpi.h"
#include <iostream> 


void forward(double *c,  const double *a){
    MPI_Comm comm = MPI_COMM_WORLD;
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if(world_rank==0)
      c[0] = a[0];
    MPI_Barrier( comm);
    MPI_Bcast( c , 1 , MPI_DOUBLE , 0 , comm);
    c[0] = pow(c[0], world_rank);
}

void backward(
  double *grad_a,
  const double *grad_c,
  const double *c, const double *a){
    MPI_Comm comm = MPI_COMM_WORLD;
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if(world_rank==0)
      grad_a[0] = 0.0;
    else
      grad_a[0] = grad_c[0] * world_rank * pow(a[0], world_rank-1);
    double ga;
    MPI_Reduce( grad_a , &ga , 1 , MPI_DOUBLE , MPI_SUM , 0 , comm);
    if(world_rank==0){
      grad_a[0] = ga;
    }
      
}
```

For implementing `m_sum`, the forward is obvious an `MPI_Reduce` function call. For the backward, we need to back-propagate the scalar gradient to each input, and this procedure requires an `MPI_Bcast` function call. 

```c++
#include "mpi.h"


void forward(double *b, const double *a){
   MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Reduce( a , b , 1 , MPI_DOUBLE , MPI_SUM , 0 , comm);
}

void backward(
    double *grad_a, const double *grad_b){
    MPI_Comm comm = MPI_COMM_WORLD;
    grad_a[0] = grad_b[0];
    MPI_Bcast( grad_a , 1 , MPI_DOUBLE , 0 , comm);
}
```

The custom operators are elegantly integrated with ADCME.jl and MPI.jl:

```julia
using MPI 
using ADCME

# interfaces to custom operators 
function basis(a)
    basis_ = load_op_and_grad("./Basis/build/libBasis","basis")
    a = convert_to_tensor(Any[a], [Float64]); a = a[1]
    basis_(a)
end

function m_sum(a)
    m_sum_ = load_op_and_grad("./Sum/build/libMSum","m_sum")
    a = convert_to_tensor(Any[a], [Float64]); a = a[1]
    m_sum_(a)
end

MPI.Init()
a = constant(2.0)
b = basis(a)
c = m_sum(b)
g = gradients(c, a)

sess = Session(); init(sess)
result = run(sess, c)
grad = run(sess, g)


if MPI.Comm_rank(MPI.COMM_WORLD)==0
    @info result, grad
end
```

To run the code, in the shell, we type
```bash
mpiexec.exe -n 4 julia test_mpi.jl
```

We obtain the outputs as expected:

```
[ Info: (15.0, 17.0)
```

## MPI Configuration in CMakeLists.txt

To configure MPI in `CMakeLists.txt`, we can simply add the following commands
```cmakelists
find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(mylib ${MPI_C_LIBRARIES})
```
