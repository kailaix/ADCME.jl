# MPI Benchmarks

The purpose of this section is to present the distributed computing capability of ADCME via MPI. With the MPI operators, ADCME is well suited to parallel implementations on clusters with very large numbers of cores. We benchmark individual operators as well as the gradient calculation as a whole. Particularly, we use **weak scaling** as the metric, i.e., how the solution time varies with the number of processors for a fixed problem size per processor. 

For most operators, ADCME is just a wrapper of existing third-party parallel computing software (e.g., Hypre). However, for gradient back-propagation, some functions may be missing and are implemented at our own discretion. For example, in Hypre, distributed sparse matrices split into multiple stripes, where each MPI rank owns a stripe with continuous row indices. In gradient back-propagation, the transpose of the original matrix is usually needed and such functionalities are missing in Hypre as of September 2020. 


## Transposition

