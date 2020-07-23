#include "common.h"

#ifdef _WIN32
#define EXPORTED __declspec(dllexport) 
#else 
#define EXPORTED 
#endif 

extern "C" EXPORTED void mpi_init(){
  int argc = 0;
  //char **argv;
  int provided;
  char**argv = {NULL};

  MPI_Init_thread( &argc , &argv , MPI_THREAD_FUNNELED, &provided);
  //MPI_Init(&argc, &argv);
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
  int world_rank;
  MPI_Comm_rank(communicator, &world_rank);
  int world_size;
  MPI_Comm_size(communicator, &world_size);

  if (world_rank == root) {
    // If we are the root process, send our data to everyone
    int i;
    for (i = 0; i < world_size; i++) {
      if (i != world_rank) {
        MPI_Send(data, count, datatype, i, 888, communicator);
      }
    }
  } else {
    // If we are a receiver process, receive the data from the root
    MPI_Recv(data, count, datatype, root, 888, communicator,
             MPI_STATUS_IGNORE);
  }
}
