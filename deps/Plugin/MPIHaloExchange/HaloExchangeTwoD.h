// M x N block 
#include "mpi.h"
#include <mutex>
#include <thread>

#define kappa(i,j) kappa[((i)-1)*(n+2)+(j)-1]
#define kappa_in(i,j) kappa_in[((i)-1)*n+(j)-1]

void HaloExchnageTwoD_forward(double *kappa, const double *kappa_in, double fill_value, int N, int M, int n, int tag){
   int r;
   MPI_Comm comm = MPI_COMM_WORLD;
   MPI_Comm_rank(MPI_COMM_WORLD, &r);
   int I = r/N + 1;
   int J = r%N + 1;
   MPI_Request requests[8];
   MPI_Status status[8];
   int n_requests = 0;
   auto left_in = new double[n];
   auto right_in = new double[n];
   auto left = new double[n];
   auto right = new double[n];
   for(int i = 1; i<=n; i++){
      for(int j=1; j<=n; j++){
         kappa(i+1, j+1) = kappa_in(i, j);
      }
   }
   if (I>1){
      MPI_Isend( kappa_in , n , MPI_DOUBLE, (I-2)*N+J-1, tag , comm , requests + n_requests);
      MPI_Irecv( kappa + 1, n , MPI_DOUBLE , (I-2)*N+J-1 , tag , comm , requests + n_requests + 1);
      n_requests += 2;
   }else{
      for(int i=2;i<=n+1;i++) 
         kappa(1, i) =  fill_value;
   }
   if (I<M){
      MPI_Isend( kappa_in + (n-1)*n, n , MPI_DOUBLE, I*N+J-1, tag , comm , requests + n_requests);
      MPI_Irecv( kappa + (n+2)*(n+1)+1, n , MPI_DOUBLE , I*N+J-1 , tag , comm , requests + n_requests + 1);
      n_requests += 2;
   }else{
      for(int i = 2; i<=n+1;i++)
         kappa(n+2, i) = fill_value;
   }
   if (J>1){
      for(int i = 1; i<=n; i++) left_in[i-1] = kappa_in(i, 1);
      MPI_Isend( left_in , n, MPI_DOUBLE, (I-1)*N+J-2, tag , comm , requests + n_requests);
      MPI_Irecv( left, n , MPI_DOUBLE, (I-1)*N+J-2, tag , comm , requests + n_requests + 1);
      n_requests += 2;
   }else{
      for(int j=2; j<=n+1; j++)
         kappa(j, 1) = fill_value;
   }
   if (J<N){
      for(int i = 1; i<=n; i++) right_in[i-1] = kappa_in(i, n);
      MPI_Isend( right_in , n, MPI_DOUBLE, (I-1)*N+J, tag , comm , requests + n_requests);
      MPI_Irecv( right, n , MPI_DOUBLE, (I-1)*N+J, tag , comm , requests + n_requests + 1);
      n_requests += 2;
   }else{
      for(int j=2;j<=n+1;j++)
         kappa(j, n+2) = fill_value;
   }
   // printf("Rank %d, tag %d, send...\n", r, tag);
   MPI_Waitall( n_requests, requests , status);
   // printf("Rank %d, tag %d, finish...\n", r, tag);
   
   if (J>1){
      for(int i = 2; i<=n+1; i++) kappa(i, 1) = left[i-2];
   }
   if (J<N){
      for(int i = 2; i<=n+1; i++) kappa(i, n+2) = right[i-2];
   }
   kappa(1, 1) = 0.0;
   kappa(n+2, 1) = 0.0;
   kappa(1, n+2) = 0.0;
   kappa(n+2, n+2) = 0.0;
   delete [] left_in;
   delete [] left;
   delete [] right_in; 
   delete [] right;
   // MPI_Barrier(comm);
}

void HaloExchnageTwoD_backward(
   double *grad_kappa_in, 
   const double *grad_kappa_, 
   const double *kappa, const double *kappa_in, double fill_value, int N, int M, int n, int tag){
   tag += 1e8;
   int r;
   MPI_Comm comm = MPI_COMM_WORLD;
   MPI_Comm_rank(MPI_COMM_WORLD, &r);
   int I = r/N + 1;
   int J = r%N + 1;
   MPI_Request requests[8];
   MPI_Status status[8];
   int n_requests = 0;
   auto left_in = new double[n];
   auto right_in = new double[n];
   auto left = new double[n];
   auto right = new double[n];
   auto up_in = new double[n];
   auto down_in = new double[n];
   
   if (I>1){
      MPI_Isend( grad_kappa_+1 , n , MPI_DOUBLE, (I-2)*N+J-1, tag , comm , requests + n_requests);
      MPI_Irecv( up_in, n , MPI_DOUBLE , (I-2)*N+J-1 , tag , comm , requests + n_requests + 1);
      n_requests += 2;
   }
   if (I<M){
      MPI_Isend( grad_kappa_ + (n+1)*(n+2)+1, n , MPI_DOUBLE, I*N+J-1, tag , comm , requests + n_requests);
      MPI_Irecv( down_in, n , MPI_DOUBLE , I*N+J-1 , tag , comm , requests + n_requests + 1);
      n_requests += 2; 
   }
   if (J>1){
      for(int i = 1; i<=n; i++) left[i-1] = grad_kappa_[ i * (n+2) ];
      MPI_Isend( left , n, MPI_DOUBLE, (I-1)*N+J-2, tag , comm , requests + n_requests);
      MPI_Irecv( left_in, n , MPI_DOUBLE, (I-1)*N+J-2, tag , comm , requests + n_requests + 1);
      n_requests += 2;
   }
   if (J<N){
      for(int i = 1; i<=n; i++) right[i-1] = grad_kappa_[ i * (n+2) + n + 1];
      MPI_Isend( right , n, MPI_DOUBLE, (I-1)*N+J, tag , comm , requests + n_requests);
      MPI_Irecv( right_in, n , MPI_DOUBLE, (I-1)*N+J, tag , comm , requests + n_requests + 1);
      n_requests += 2;
   }

   // printf("BACKWORD Rank %d, tag %d, send...\n", r, tag);   
   MPI_Waitall( n_requests, requests , status);
   // printf("BACKWORD Rank %d, tag %d, finish...\n", r, tag);

   if (I>1){
      for (int i = 0; i<n;i++) grad_kappa_in[i] += up_in[i];
   }
   if (I<M){
      for (int i = 0; i<n;i++) grad_kappa_in[(n-1)*n+i] += down_in[i];
   }
   if (J>1){
      for(int i = 1; i<=n; i++) grad_kappa_in[(i-1)*n] += left_in[i-1];
   }
   if (J<N){
      for(int i = 1; i<=n; i++) grad_kappa_in[(i-1)*n + n-1] += right_in[i-1];
   }
   delete [] left_in;
   delete [] left;
   delete [] right_in; 
   delete [] right;
   delete [] up_in;
   delete [] down_in;

   for(int i = 1; i<=n; i++){
      for(int j=1; j<=n; j++){
         grad_kappa_in[(i-1)*n+j-1] += grad_kappa_[i*(n+2)+j];
      }
   }   

}


#undef kappa
#undef kappa_in
