void forward(double *out, const double *u, const double *up, const double *down, 
          const double*left, const double *right, const double *f, int m, int n, double h){
  for(int i = 1; i<n-1;i++)
    for(int j = 1;j<m-1;j++){
      out[i*m+j] = (u[(i+1)*m+j] + u[(i-1)*m+j] + u[i*m+j+1] + u[i*m+j-1] - h * h * f[i*m+j])/4;
    }
      
  int i, j;
  for(int i = 1;i<n-1;i++){
    j = 0;
    out[i*m+j] = (u[(i+1)*m+j] + u[(i-1)*m+j] + u[i*m+j+1] + left[i] - h * h * f[i*m+j])/4;
    j = m-1;
    out[i*m+j] = (u[(i+1)*m+j] + u[(i-1)*m+j] + right[i] + u[i*m+j-1]  - h * h * f[i*m+j])/4;
  }

  for(int j = 1;j<m-1;j++){
    i = 0;
    out[i*m+j] = (u[(i+1)*m+j] + up[j] + u[i*m+j+1] + u[i*m+j-1] - h * h * f[i*m+j])/4;
    i = n-1;
    out[i*m+j] = (down[j] + u[(i-1)*m+j] + u[i*m+j+1] + u[i*m+j-1] - h * h * f[i*m+j])/4;
  }

  out[0] = (up[0] + left[0] + u[1] + u[m] - h * h * f[0])/4;
  out[m-1] = (up[m-1] + right[0] + u[m-2] + u[2*m-1] - h * h * f[m-1])/4;
  out[(n-1)*m] = (left[n-1] + down[0] + u[(n-1)*m+1] + u[(n-2)*m] - h * h * f[(n-1)*m])/4;
  out[n*m-1] = (right[n-1] + down[m-1] + u[n*m-2] + u[(n-1)*m-1] - h * h * f[n*m-1])/4;
}

void data_forward(double *out_left, double *out_right, 
    double *out_up, double *out_down, const double *left, const double *right, 
    const double *up, const double *down, int M, int N, int m, int n){
      int I, J;
      MPI_Comm comm = MPI_COMM_WORLD;
      int rank;
      MPI_Comm_rank( comm , &rank);
      I = rank/M; J = rank%M;

      // printf(">>>> I = %d, J = %d\n", I, J);

      MPI_Request	r1, r2, r3, r4, r5, r6, r7, r8;
      if (I>0)
        MPI_Isend( up , m , MPI_DOUBLE , (I-1)*M+J , 0, comm , &r1);

      if (I<N-1)
        MPI_Isend( down , m , MPI_DOUBLE , (I+1)*M+J , 0 , comm , &r2);

      if (J>0)
        MPI_Isend( left , n , MPI_DOUBLE , I*M+J-1 , 0 , comm , &r3);

      if (J<M-1)
        MPI_Isend( right , n , MPI_DOUBLE , I*M+J+1 , 0 , comm , &r4);

      if (I>0)
        MPI_Irecv( out_up , m , MPI_DOUBLE , (I-1)*M+J , 0 , comm , &r5);
      
      if (I<N-1)
        MPI_Irecv( out_down , m , MPI_DOUBLE, (I+1)*M+J, 0, comm, &r6);
      
      if (J>0)
        MPI_Irecv( out_left , n , MPI_DOUBLE , I*M+J-1 , 0 , comm , &r7);

      if (J<M-1)
        MPI_Irecv( out_right , n , MPI_DOUBLE , I*M+J+1 , 0 , comm , &r8);

      MPI_Status status;
      if(I>0) MPI_Wait( &r1, &status);
      if(I<N-1) MPI_Wait( &r2, &status);
      if(J>0) MPI_Wait( &r3, &status);
      if(J<M-1) MPI_Wait( &r4, &status);
      if(I>0) MPI_Wait( &r5, &status);
      if(I<N-1) MPI_Wait( &r6, &status);
      if(J>0) MPI_Wait( &r7, &status);
      if(J<M-1) MPI_Wait( &r8, &status);

      // printf("I = %d, J = %d\n", I, J);

  }