#include <mpi.h>
#include <iostream>

#ifdef _WIN32
#define EXPORTED __declspec(dllexport)
#else
#define EXPORTED
#endif 

extern "C" EXPORTED int printinfo(){
    MPI_Comm comm = MPI_COMM_WORLD;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[1000];
    int name_len;
    MPI_Get_processor_name( processor_name , &name_len);
    printf("Hello world from processor %s, rank %d out of %d processors\n",
        processor_name, world_rank, world_size);
    int result;
    MPI_Reduce( &world_rank , &result , 1 , MPI_INT , MPI_SUM , 0 , comm);
    return result;
}