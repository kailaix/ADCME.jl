#include "mpi.h"
#include <thread>

void forward(double *b, const double *a){
   MPI_Comm comm = MPI_COMM_WORLD;
   std::cout << "MSUM thread "<<std::this_thread::get_id() << std::endl;
    MPI_Reduce( a , b , 1 , MPI_DOUBLE , MPI_SUM , 0 , comm);
}

void backward(
    double *grad_a, const double *grad_b){
    MPI_Comm comm = MPI_COMM_WORLD;
    grad_a[0] = grad_b[0];
    MPI_Bcast( grad_a , 1 , MPI_DOUBLE , 0 , comm);
}