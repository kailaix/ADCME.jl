#include "common.h"

#ifdef _WIN32
#define EXPORTED __declspec(dllexport) 
#else 
#define EXPORTED 
#endif 

extern "C" EXPORTED void mpi_init(){
  int argc = 0;
  int provided;
  char**argv = {NULL};

  MPI_Init_thread( NULL, NULL , MPI_THREAD_MULTIPLE, &provided);
}

extern "C" EXPORTED void mpi_finalize(){
  MPI_Finalize();
}

extern "C" EXPORTED int mpi_rank(){
  int rank;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank( comm , &rank);
  return rank;
}

extern "C" EXPORTED int mpi_size(){
  int size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_size( comm , &size);
  return size;
}

extern "C" EXPORTED bool mpi_finalized(){
  int flag;
  MPI_Finalized(&flag);
  return flag;
}

extern "C" EXPORTED bool mpi_initialized(){
  int flag;
  MPI_Initialized(&flag);
  return flag;
}

extern "C" EXPORTED void mpi_sync(long long* data, int count, int root){
  MPI_Comm comm = MPI_COMM_WORLD;
  int world_rank;
  MPI_Comm_rank(comm, &world_rank);
  int world_size;
  MPI_Comm_size(comm, &world_size);
  MPI_Request request;
  MPI_Status status; 
  MPI_Ibcast(data, count, MPI_LONG_LONG, root, comm, &request);
  MPI_Wait( &request , &status);  
}

extern "C" EXPORTED void mpi_sync_double(double* data, int count, int root){
  MPI_Comm comm = MPI_COMM_WORLD;
  int world_rank;
  MPI_Comm_rank(comm, &world_rank);
  int world_size;
  MPI_Comm_size(comm, &world_size);
  MPI_Request request;
  MPI_Status status; 
  MPI_Ibcast(data, count, MPI_DOUBLE, root, comm, &request);
  MPI_Wait( &request , &status);  
}