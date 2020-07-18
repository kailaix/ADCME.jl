#include "common.h"

#ifdef _WIN32
#define EXPORTED __declspec(dllexport) 
#else 
#define EXPORTED 
#endif 

extern "C" EXPORTED void mpi_init(){
  int argc;
  char **argv;
  int provided;
  MPI_Init_thread( &argc , &argv , MPI_THREAD_FUNNELED , &provided);
  // MPI_Init(&argc, &argv);
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

extern "C" EXPORTED void mpi_sync(long long* value, int n, int root){
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Bcast( value , n , MPI_LONG_LONG_INT , root , comm);
}