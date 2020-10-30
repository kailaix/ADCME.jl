# Understand the Multi-threading Model 


Multi-threading is a very important technique for accelerating simulations in ADCME. Through this section, we look into the multi-threading models of ADCME's backend, TensorFlow. Let us start with some basic concepts related to CPUs.

## Processes, Threads, and Cores

We often hear about processes and threads when talking about multi-threading. In a word, a process is a program in execution. A process may invoke multiple threads. A thread can be viewed as a scheduler for executing each line of codes in order. It tells the CPU to perform an instruction. 

The biggest difference is that different processes do not share memory with each other. But different threads **within the same process** has the same memory space. 

Up till now, processes and threads are **logical** concepts, which are not bound to the physical devices. Cores are physical concepts. Nowadays, each CPU contains multiple cores. Cores contain workers that actually perform computation, and these workers are called ALUs (arithmetic logic units). To use these ALUs, we need some schedulers that tell the CPU/cores to perform certain tasks. This is done by **hardware threads**. Typically when we want to do some computation, we need to load data first and then perform calculations on ALUs. The data loading process is also taken care by the hardware threads. Therefore, it is possible that the ALUs are waiting for input data. One clever idea in computer science is to use pipelining: overlapping data loading of the current instruction and computation of the last instruction. This means we need more schedulers, i.e., hardware threads, to take care of data loading and computing simultaneously. Therefore, modern CPUs usually have multiple hardware threads for one core. For example, Intel CPUs have the so-called **hyperthreading technology**, i.e., each core has two physical threads. 

Now we understand four concepts: processes, (logical) threads, cores, hardware threads. So what is the relationship between threads and hardware threads? Actually this is straight-forward: logical threads are mapped to hardware threads. For example, if there are 4 logical threads and 4 hardware threads, the OS may map each logical thread to one distinct hardware thread. If there are more than 4 logical threads, some logical threads may be mapped to one. That says, these logical threads will not enjoy truly parallelism.

Now let's consider how ADCME works: for one CPU, ADCME always runs only one process. But to gain maximum efficiency, ADCME will create multiple threads to leverage any parallelism we have in hardware resources and computational models. 

## Inter and Intra Parallelism

There are two types of parallelism in ADCME execution: **inter** and **intra**. 

Consider a computational graph, there may be multiple independent operators and therefore we can execute them in parallel. This type of parallelism is called inter-parallelism. For example, 

```julia
using ADCME 

a1 = constant(rand(10))
a2 = constant(rand(10))
a3 = a1 * a2 
a4 = a1 + a2 
a5 = a3 + a4
```

In the above code, `a3 = a1 * a2` and `a4 = a1 + a2` are independent and can be executed in parallel. 

Another type of parallelism is intra-parallel, that is, the computation within each operator can be computed in parallel. For example, in the example above, we can compute the first 5 entries and last 5 entries in `a4 = a1 + a2` in parallel. 

These types of parallelism can be achieved using multi-threading. In the next section, we explain how this is implemented in TensorFlow.

## ThreadPools

The backend of ADCME, TensorFlow, uses two threadpools for multithreading. One thread pool is for inter-parallelism, and the other is for intra-parallelism. They can be set by the users.

One common mistake is that users think in terms of hardware threads instead of logical threads. For example, if we have 8 hardware threads in total, one might want to allocate 4 threads for inter-parallelism and 4 threads for intra-parallelism. That's not true. When we talk about allocating threads for intra and inter thread pools, we are always talking about logical threads. So we can have a threadpool containing 100 threads for both inter and intra thread pools even if we only have 8 hardware threads in total. And in the runtime, inter and intra thread pools may use the same hardware threads for scheduling tasks. 

The following figure is an illustration of the two thread pools of ADCME. 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/threadpool.png?raw=true)

## How to Use the Intra Thread Pool

In practice, when we implement custom operators, we may want to use the intra thread pool. [Here](https://github.com/kailaix/ADCME.jl/tree/master/docs/src/assets/Codes/ThreadPools) gives an example how to use thread pools. 

```c++
#include <thread>
#include <chrono>
#include <condition_variable>
#include <atomic>

void print_thread(std::atomic_int &cnt, std::condition_variable &cv){
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  printf("My thread ID is %d\n", std::this_thread::get_id());
  cnt++;
  if (cnt==7) cv.notify_one();
}

void threadpool_print(OpKernelContext* context){
  thread::ThreadPool * const tp = context->device()->tensorflow_cpu_worker_threads()->workers;
  std::atomic_int cnt = 0;
  std::condition_variable cv;
  std::mutex mu;

  printf("Number of intra-parallel thread = %d\n", tp->NumThreads());
  printf("Maximum Parallelism = %d\n", port::MaxParallelism());

  for (int i = 0; i < 7; i++)
    tp->Schedule([&cnt, &cv](){print_thread(cnt, cv);});
  
  {
    std::unique_lock<std::mutex> lck(mu);
    cv.wait(lck, [&cnt](){return cnt==7;});
  }
  printf("Op finished\n");
}
```

Basically, we can asynchronously launch jobs using the thread pools. Additionally, we are responsible for synchronization. Here we have used condition variables for synchronization. 

Typically our CPU operators are synchronous and do not need the thread pools. But it does not hard to have an intra thread pool. 


## Runtime Optimizations

If you are using Intel CPUs, we may have some runtime optimization configurations. See this [link](https://software.intel.com/content/www/us/en/develop/articles/guide-to-tensorflow-runtime-optimizations-for-cpu.html) for details. Here, we show the effects of some optimizations. 

We already understand `intra_op_parallelism_threads` and `inter_op_parallelism_threads`; now let us consider some other options. We consider computing $\sin$ function using the following formula 

$$\sin x \approx x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!}$$

The implementation can be found [here]().

### Configure OpenMP 

To set the number of OMP threads, we can configure the `OMP_NUM_THREADS` environment variable. One caveat is that the variable must be set before loading ADCME. For example 

```julia
ENV["OMP_NUM_THREADS"] = 5
using ADCME
```

Running the `omp_thread.jl`, we have the following output

```
There are 5 OpenMP threads
4 is computing...
0 is computing...
4 is computing...
1 is computing...
1 is computing...
0 is computing...
3 is computing...
3 is computing...
2 is computing...
2 is computing...
```

We see that there are 5 threads running. 

### Configure Number of Devices


[`Session`](@ref) accepts keywords `CPU`, which limits the number of CPUs we can use. Note, `CPU` corresponds to the number of CPU devices, not cores or threads. For example, if we run `num_device.jl` with (default is using all CPUs)

```julia
sess = Session(CPU=1); init(sess)
```

We will see 
```
There are 144 OpenMP threads
```
This is because we have 144 cores in our machine. 
