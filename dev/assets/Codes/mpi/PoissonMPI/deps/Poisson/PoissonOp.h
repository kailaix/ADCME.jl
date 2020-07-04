#include "mpi.h"

int M = POISSON_M;
int N = POISSON_N;

static double t_send_recv = 0.0, t_inner = 0.0, t_boundary = 0.0;
int mpi_cnt = 0;

// u is a (m+2)*(n+2) matrix. 
// out_left, out_right, out_up, out_down
// in_...
void PoissonOpOp::forward(double *ut, const double *u, const double *f, double h, int m, int n){

  double timer1, timer2;
  timer1 = MPI_Wtime();


  MPI_Comm comm = MPI_COMM_WORLD;
  int my_rank, dest;
  MPI_Comm_rank( comm , &my_rank);
  int I = my_rank/N, J = my_rank%N;
  // printf("I = %d, J = %d, m = %d, n = %d, M = %d, N = %d\n", I, J, m, n, M, N);
  // step 1: send boundary values
  if (I>0) {
    dest = (I-1)*N + J;
    // printf("Send value = %f %f %f\n", u[0], u[1], u[2]);
    MPI_Request mr;
    MPI_Isend( u , n , MPI_DOUBLE ,  dest, 0 , comm, &mr);
  }

  if (I<M-1){
    dest = (I+1)*N + J;
    MPI_Request mr;
    MPI_Isend( u+(m-1)*n, n, MPI_DOUBLE, dest, 0, comm, &mr);
  }

  if (J>0){
    dest = I*N+J-1;
    for(int i = 0; i<m; i++) out_left[i] = u[i*n];
    MPI_Request mr;
    MPI_Isend( out_left, m, MPI_DOUBLE, dest, 0, comm, &mr);
  }
  
  if (J<N-1){
    dest = I*N+J+1;
    for(int i = 0; i<m; i++) out_right[i] = u[i*n+n-1];
    MPI_Request mr;
    MPI_Isend( out_right, m, MPI_DOUBLE, dest, 0, comm, &mr);
  }

  // printf("Boundary values sent!\n");

  // step 2: receive values from all directions
  MPI_Status status;
  MPI_Request mr1, mr2, mr3, mr4;
  bool set1 = false, set2 = false, set3 = false, set4 = false;
  if (I>0){
    dest = (I-1)*N + J;
    MPI_Irecv( in_up , n , MPI_DOUBLE , dest , 0 , comm , &mr1);
    set1 = true;
  }

  if (I<M-1){
    dest = (I+1)*N + J;
    MPI_Irecv( in_down , n , MPI_DOUBLE , dest , 0 , comm , &mr2);
    set2 = true;
    // printf("Received value = %f %f %f\n", in_down[0], in_down[1], in_down[2]);
  }

  if (J>0){
    dest = I*N + J-1;
    set3 = true;
    MPI_Irecv( in_left , m , MPI_DOUBLE , dest , 0 , comm , &mr3);
  }

  if (J<N-1){
    dest = I*N + J+1;
    set4 = true;
    MPI_Irecv( in_right , m , MPI_DOUBLE , dest , 0 , comm , &mr4);
  }

  

  // printf("Boundary values received!\n");

  timer2 = MPI_Wtime();
  // step 3: update ut 
  for(int i=1;i<m-1;i++)
    for(int j=1;j<n-1;j++)
      ut[i*n+j] = (
        u[(i-1)*n+j] + u[(i+1)*n+j] + u[i*n+j-1] + u[i*n+j+1] - h*h*f[i*n+j]
      )/4;

  t_inner += MPI_Wtime() - timer2;

  if (set1) MPI_Wait( &mr1 , &status);
  if (set2) MPI_Wait( &mr2 , &status);
  if (set3) MPI_Wait( &mr3 , &status);
  if (set4) MPI_Wait( &mr4 , &status);

  t_send_recv += MPI_Wtime() - timer1;

  timer1 = MPI_Wtime();

    
  ut[0] = (in_up[0] + in_left[0] + u[1] + u[n] - h*h*f[0])/4;
  ut[n-1] = (in_up[n-1] + in_right[0] + u[n-2] + u[2*n-1] - h*h*f[n-1])/4;
  ut[(m-1)*n] = (in_left[m-1] + in_down[0] + u[(m-1)*n+1] + u[(m-2)*n] - h*h*f[(m-1)*n])/4;
  ut[m*n-1] = (in_right[m-1] + in_down[n-1] + u[m*n-2] + u[(m-1)*n-1] - h*h*f[m*n-1])/4;

  for(int j=1;j<n-1;j++) {
    ut[j] = (in_up[j] + u[j-1] + u[j+1] + u[j+n] - h*h*f[j])/4;
  }

  for(int j=1;j<n-1;j++) {
    ut[(m-1)*n+j] = (in_down[j] + u[(m-1)*n+j-1] + u[(m-1)*n+j+1] + u[(m-1)*n+j-n] - h*h*f[(m-1)*n+j])/4;
  }

  for(int i=1;i<m-1;i++){
    ut[i*n] = (u[(i-1)*n] + u[(i+1)*n] + in_left[i] + u[i*n+1] - h*h*f[i*n])/4;
  }

  for(int i=1;i<m-1;i++){
    ut[i*n+n-1] = (u[(i-1)*n + n-1] + in_right[i] + u[(i+1)*n+n-1] + u[i*n+n-2] - h*h*f[i*n+n-1])/4;
  }

  t_boundary += MPI_Wtime() - timer1;
  mpi_cnt += 1;
  // printf("Operator finished!\n");

}


extern "C" void report_time(){
   MPI_Comm comm = MPI_COMM_WORLD;
  int my_rank, dest;
  MPI_Comm_rank( comm , &my_rank);
  int I = my_rank/N, J = my_rank%N;
  double total = t_boundary + t_send_recv;
  printf("[%d %d --> %d] t_total = %f, t_inner = %f, t_boundary = %f, t_send_recv = %f\n", I, J, mpi_cnt, total, t_inner, t_boundary, t_send_recv);
}