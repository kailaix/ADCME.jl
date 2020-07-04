#include "mpi.h"
#include <iostream> 
#include <thread>

void forward(double *c,  const double *a){
  std::cout << "BASIS thread "<<std::this_thread::get_id() << std::endl;
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