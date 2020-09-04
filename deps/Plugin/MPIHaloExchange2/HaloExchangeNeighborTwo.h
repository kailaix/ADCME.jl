#include "mpi.h"
#include <memory>
using std::shared_ptr;

#define U_OUT(i,j) U_OUT[(i)*(n+4)+(j)]
#define U_IN(i,j) U_IN[(i)*n+(j)]
#define GRAD_U_OUT(i,j) GRAD_U_OUT[(i)*(n+4)+(j)]
#define GRAD_U_IN(i,j) GRAD_U_IN[(i)*n+(j)]
#define RANK(I,J) ((I-1)*N+(J-1))

void HaloExchangeNeighborTwo_forward(double *U_OUT, const double *U_IN, double fill_value,
   int M, int N, int n, int tag){
  int rank;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank( comm , &rank);
  int I = rank/N + 1;
  int J = rank%N + 1;
  
  #pragma omp parallel for 
  for(int i = 0; i<n; i++){
    for(int j = 0; j<n; j++){
      U_OUT(i+2, j+2) = U_IN(i, j);
    }
  }

  shared_ptr<double> iupper_in(new double[2*n]);
  shared_ptr<double> ilower_in(new double[2*n]);
  shared_ptr<double> jupper_in(new double[2*n]);
  shared_ptr<double> jlower_in(new double[2*n]);
  shared_ptr<double> iupper(new double[2*n]);
  shared_ptr<double> ilower(new double[2*n]);
  shared_ptr<double> jupper(new double[2*n]);
  shared_ptr<double> jlower(new double[2*n]);

  MPI_Request request[8];
  MPI_Status status[8];
  int n_request = 0;

  for(int k = 0; k<n; k++){
    iupper_in.get()[k] = U_IN(0, k);
    iupper_in.get()[n + k] = U_IN(1, k);

    ilower_in.get()[k] = U_IN(n-2, k);
    ilower_in.get()[n + k] = U_IN(n-1, k);
    
    jupper_in.get()[k] = U_IN(k, 0);
    jupper_in.get()[n + k] = U_IN(k, 1);
    
    jlower_in.get()[k] = U_IN(k, n-2);
    jlower_in.get()[n + k] = U_IN(k, n-1);
  }

  if (I>1){
    MPI_Isend( iupper_in.get() , 2*n , MPI_DOUBLE , RANK(I-1, J) , tag , comm , request + n_request);
    MPI_Irecv( iupper.get() , 2*n , MPI_DOUBLE , RANK(I-1,J) , tag , comm , request + n_request+1);
    n_request += 2;
  } else{
    for(int i = 0; i< 2*n; i++) iupper.get()[i] = fill_value;
  }

  if (I<M){
    MPI_Isend( ilower_in.get() , 2*n , MPI_DOUBLE , RANK(I+1, J) , tag , comm , request + n_request);
    MPI_Irecv( ilower.get() , 2*n , MPI_DOUBLE , RANK(I+1,J) , tag , comm , request + n_request+1);
    n_request += 2;
  } else{
    for(int i = 0; i< 2*n; i++) ilower.get()[i] = fill_value;
  }

  if (J>1){
    MPI_Isend( jupper_in.get() , 2*n , MPI_DOUBLE , RANK(I, J-1) , tag , comm , request + n_request);
    MPI_Irecv( jupper.get() , 2*n , MPI_DOUBLE , RANK(I,J-1) , tag , comm , request + n_request+1);
    n_request += 2;
  }else{
    for(int i = 0; i< 2*n; i++) jupper.get()[i] = fill_value;
  }

  if (J<N){
    MPI_Isend( jlower_in.get() , 2*n , MPI_DOUBLE , RANK(I, J+1) , tag , comm , request + n_request);
    MPI_Irecv( jlower.get() , 2*n , MPI_DOUBLE , RANK(I,J+1) , tag , comm , request + n_request+1);
    n_request += 2;
  }else{
    for(int i = 0; i< 2*n; i++) jlower.get()[i] = fill_value;
  }

  if (n_request>0){
    MPI_Waitall( n_request , request , status);
  }
  for(int k = 0; k<n; k++){
    U_OUT(0, k+2) = iupper.get()[k];
    U_OUT(1, k+2) = iupper.get()[n + k];
    
    U_OUT(k+2, 0) = jupper.get()[k];
    U_OUT(k+2, 1) = jupper.get()[n + k];

    U_OUT(n+2, k+2) = ilower.get()[k];
    U_OUT(n+3, k+2) = ilower.get()[n + k];
    
    U_OUT(k+2, n+2) = jlower.get()[k];
    U_OUT(k+2, n+3) = jlower.get()[n + k];
  }


}



void HaloExchangeNeighborTwo_backward(
  double *GRAD_U_IN,
  const double *GRAD_U_OUT,  
  const double *U_OUT, const double *U_IN, double fill_value,
   int M, int N, int n, int tag){
  int rank;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank( comm , &rank);
  int I = rank/N + 1;
  int J = rank%N + 1;
  
  #pragma omp parallel for 
  for(int i = 0; i<n; i++){
    for(int j = 0; j<n; j++){
      GRAD_U_IN(i, j) = GRAD_U_OUT(i+2, j+2);
    }
  }

  shared_ptr<double> iupper_in(new double[2*n]);
  shared_ptr<double> ilower_in(new double[2*n]);
  shared_ptr<double> jupper_in(new double[2*n]);
  shared_ptr<double> jlower_in(new double[2*n]);
  shared_ptr<double> iupper(new double[2*n]);
  shared_ptr<double> ilower(new double[2*n]);
  shared_ptr<double> jupper(new double[2*n]);
  shared_ptr<double> jlower(new double[2*n]);

  MPI_Request request[8];
  MPI_Status status[8];
  int n_request = 0;

  for(int k = 0; k<n; k++){
    iupper_in.get()[k] = GRAD_U_OUT(0, k+2);
    iupper_in.get()[n + k] = GRAD_U_OUT(1, k+2);

    ilower_in.get()[k] = GRAD_U_OUT(n-2+4, k+2);
    ilower_in.get()[n + k] = GRAD_U_OUT(n-1+4, k+2);
    
    jupper_in.get()[k] = GRAD_U_OUT(k+2, 0);
    jupper_in.get()[n + k] = GRAD_U_OUT(k+2, 1);
    
    jlower_in.get()[k] = GRAD_U_OUT(k+2, n-2+4);
    jlower_in.get()[n + k] = GRAD_U_OUT(k+2, n-1+4);
  }

  if (I>1){
    MPI_Isend( iupper_in.get() , 2*n , MPI_DOUBLE , RANK(I-1, J) , tag , comm , request + n_request);
    MPI_Irecv( iupper.get() , 2*n , MPI_DOUBLE , RANK(I-1,J) , tag , comm , request + n_request+1);
    n_request += 2;
  } else{
    for(int i = 0; i< 2*n; i++) iupper.get()[i] = 0.0;
  }

  if (I<M){
    MPI_Isend( ilower_in.get() , 2*n , MPI_DOUBLE , RANK(I+1, J) , tag , comm , request + n_request);
    MPI_Irecv( ilower.get() , 2*n , MPI_DOUBLE , RANK(I+1,J) , tag , comm , request + n_request+1);
    n_request += 2;
  } else{
    for(int i = 0; i< 2*n; i++) ilower.get()[i] = 0.0;
  }

  if (J>1){
    MPI_Isend( jupper_in.get() , 2*n , MPI_DOUBLE , RANK(I, J-1) , tag , comm , request + n_request);
    MPI_Irecv( jupper.get() , 2*n , MPI_DOUBLE , RANK(I,J-1) , tag , comm , request + n_request+1);
    n_request += 2;
  }else{
    for(int i = 0; i< 2*n; i++) jupper.get()[i] = 0.0;
  }

  if (J<N){
    MPI_Isend( jlower_in.get() , 2*n , MPI_DOUBLE , RANK(I, J+1) , tag , comm , request + n_request);
    MPI_Irecv( jlower.get() , 2*n , MPI_DOUBLE , RANK(I,J+1) , tag , comm , request + n_request+1);
    n_request += 2;
  }else{
    for(int i = 0; i< 2*n; i++) jlower.get()[i] = 0.0;
  }

  if (n_request>0){
    MPI_Waitall( n_request , request , status);
  }
    

  for(int k = 0; k<n; k++){
    GRAD_U_IN(0, k) += iupper.get()[k];
    GRAD_U_IN(1, k) += iupper.get()[n + k];
    
    GRAD_U_IN(k, 0) += jupper.get()[k];
    GRAD_U_IN(k, 1) += jupper.get()[n + k];

    GRAD_U_IN(n-2, k) += ilower.get()[k];
    GRAD_U_IN(n-1, k) += ilower.get()[n + k];
    
    GRAD_U_IN(k, n-2) += jlower.get()[k];
    GRAD_U_IN(k, n-1) += jlower.get()[n + k];
  }


}


#undef U_OUT
#undef U_IN
#undef GRAD_U_OUT
#undef GRAD_U_IN
#undef RANK