# MPI Benchmarks

The purpose of this section is to present the distributed computing capability of ADCME via MPI. With the MPI operators, ADCME is well suited to parallel implementations on clusters with very large numbers of cores. We benchmark individual operators as well as the gradient calculation as a whole. Particularly, we use **weak scaling** as the metric, i.e., how the solution time varies with the number of processors for a fixed problem size per processor. 

For most operators, ADCME is just a wrapper of existing third-party parallel computing software (e.g., Hypre). However, for gradient back-propagation, some functions may be missing and are implemented at our own discretion. For example, in Hypre, distributed sparse matrices split into multiple stripes, where each MPI rank owns a stripe with continuous row indices. In gradient back-propagation, the transpose of the original matrix is usually needed and such functionalities are missing in Hypre as of September 2020. 

The MPI programs are verified with serial programs. 

## Transposition


The following results shows the scalability of the transposition operator of [`mpi_SparseTensor`](@ref). In Hypre, each MPI processor own a set of rows of the whole sparse matrix. The rows have continuous indices. To transpose the sparse matrix in a parallel environment, we first split the matrices in each MPI processor into blocks and then use `MPI_Isend`/`MPI_Irecv` to exchange data. Finally, we transpose the matrices in place for each block. Using this method, we obtained a CSR representation of the transposed matrix. 


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mpi/mpi_transpose.png?raw=true)

In the following plots, we show the strong scaling for a fixed matrix size of $25\text{M} \times 25\text{M}$ as well as the weak scaling, where each MPI processor owns $300^2=90000$ rows. 

| Strong Scaling | Weak Scaling |
|----------------|--------------|
|    ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mpi/transpose_strong.png?raw=true)            |      ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mpi/transpose_weak.png?raw=true)        |

## Poisson's Equation

This example presents the overhead of ADCME MPI operators when the main computation is carried out through a third-party library (Hypre). We solve the Poisson's equation 

$$\nabla \cdot (\kappa(x, y) \nabla u(x,y)) = f(x, y), (x,y) \in \Omega \quad u(x,y) = 0, (x,y) \in \partial \Omega$$

$\kappa(x, y)$ is approximated by a neural network $\kappa_\theta(x,y) = \mathcal{NN}_\theta(x,y)$, and the weights and biases $\theta$ are broadcasted from the root processor. We express the numerical scheme as a computational graph is:

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mpi/poisson_cg.png?raw=true) 



We show the strong scaling with a fixed problem size $1800 \times 1800$ (mesh size, which implies the matrix size is around 32 million). We also show the weak scaling where each MPI processor owns a $300\times 300$ block. For example, a problem with 3600 processors has the problem size $90000\times 3600 \approx 0.3$ billion.

### Weak Scaling

We first consider the weak scaling. The following plots shows the runtime for forward computation as well as gradient back-propagation. 

| Speedup | Efficiency |
|----------------|--------------|
|    ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mpi/time_core1.png?raw=true)            |      ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mpi/efficiency_core14.png?raw=true)        |

There are several important observations:

1. By using more cores per processor, the runtime is reduced significantly. For example, the runtime for the backward is reduced to around 10 seconds from 30 seconds by switching from 1 core to 4 cores per processor. 
2. The runtime for the backward is typically less than twice the forward computation. Although the backward requires solve two linear systems (one of them is in the forward computation), the AMG linear solver in the back-propagation may converge faster, and therefore costs less than the forward. 

Additionally, we show the overhead, which is defined as the difference between total runtime and Hypre linear solver time, of both the forward and backward calculation. 

| 1 Core | 4 Cores |
|----------------|--------------|
|    ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mpi/speedup_core14.png?raw=true)            |      ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mpi/overhead_core4.png?raw=true)        |

We see that the overhead is quite small compared to the total time, especially when the problem size is large. This indicates that the ADCME MPI implementation is very effective. 

### Strong Scaling

Now we consider the strong scaling. In this case, we fixed the whole problem size and split the mesh onto different MPI processors. The following plots show the runtime for the forward computation and the gradient back-propagation

| Forward | Bckward |
|----------------|--------------|
|    ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mpi/serial_forward.png?raw=true)            |      ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mpi/serial_backward.png?raw=true)        |


The following plots show the speedup and efficiency 


| 1 Core | 4 Cores |
|----------------|--------------|
|    ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mpi/time_core1.png?raw=true)            |      ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/mpi/time_core4.png?raw=true)        |


We can see that the 4 cores have smaller runtime compared to 1 core. 



## Acoustic Seismic Inversion

## Elastic Seismic Inversion



