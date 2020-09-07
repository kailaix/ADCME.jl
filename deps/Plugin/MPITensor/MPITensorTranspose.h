#include "mpi.h"
#include <vector>
#include <memory>

using std::vector;

namespace MPITensorTranspose{
  class Block{
    public:
      vector<int> row;
      vector<int> col;
      vector<double> val;
      std::vector<int> orow;
      std::vector<int> ocol;
      std::vector<double> oval;
      MPI_Comm comm;
      int nrows;

      MPI_Request req_recv_n, req_send_n, req_send_data[3], req_recv_data[3];

      int recv_n;
      int index, rank;
      void add(int i, int j, double v){
        row.push_back(i);
        col.push_back(j);
        val.push_back(v);
        // printf("block #%d, row = %d, col = %d, val = %f\n", index, i, j, v);
      }
      void send_size(){
        int n = row.size();
        MPI_Isend( &n , 1 , MPI_INT , index , 0 , comm, &req_send_n);
        MPI_Irecv( &recv_n , 1 , MPI_INT , index , 0 , comm , &req_recv_n);
      }

      void allocate_memory(){
        orow.resize(recv_n);
        ocol.resize(recv_n);
        oval.resize(recv_n);
        // printf("Index %d, allocate length %d array\n", index, recv_n);
      }

      int get_physical_col(int idx){
          return ocol[idx] + index * nrows;
      }

      void send_data(){
        int n = row.size();
        MPI_Isend( row.data(), n, MPI_INT, index, 0, comm, req_send_data);
        MPI_Isend( col.data(), n, MPI_INT, index, 1, comm, req_send_data+1);
        MPI_Isend( val.data(), n, MPI_DOUBLE, index, 2, comm, req_send_data+2);

        MPI_Irecv( ocol.data(), recv_n, MPI_INT, index, 0, comm, req_recv_data);
        MPI_Irecv( orow.data(), recv_n, MPI_INT, index, 1, comm, req_recv_data+1);
        MPI_Irecv( oval.data(), recv_n, MPI_DOUBLE, index, 2, comm, req_recv_data+2);
      }


      Block(int rank, int index, int nrows): index(index), rank(rank), nrows(nrows){
        comm = MPI_COMM_WORLD;
      };
  };

  class Forward{
    public:
      vector<Block> blocks;
      int n_blocks;
      int size(){
        int n = 0;
        for (int i = 0; i<n_blocks; i++) n += blocks[i].recv_n;
        return n;
      }
      void copy(int64 *indices, double *val);
      Forward(int nrows, int mat_size, int rank, 
            const int *row, int n_row, const int *col, const int *ncol, const double *val);

  };

  
Forward::Forward(int nrows, int mat_size, int rank, 
            const int *row, int n_row, const int *col, const int *ncol, const double *val){
      // printf("nrows = %d, mat_size = %d, rank = %d, n_row = %d\n", nrows, mat_size, rank, n_row);
      n_blocks = mat_size/nrows;
      for(int i = 0; i<n_blocks; i++) blocks.emplace_back(rank, i, nrows);
      int ilower = rank * nrows;
      int k = 0;
      for (int i = 0; i<n_row; i++){
        
        int ncol_ = ncol[i];
        int row_ = row[i] - ilower;
        if (ncol[i]==0) continue; 
        for(int z = 0; z<ncol[i]; z++){
          int col_ = col[k];
          double val_ = val[k];
          k++;
          int index = col_/nrows;
          // printf("%d, %d --> %d, val = %f, val_= %f\n", row_, col_, col_ - (col_/nrows) * nrows, val[k-1], val_);
          col_ = col_ - (col_/nrows) * nrows;
          blocks[index].add(row_, col_, val_);
        }

      }

      // send sizes
      for (int i = 0; i<n_blocks; i++) blocks[i].send_size();

      MPI_Status status[3];
      for (int i = 0; i<n_blocks; i++){
        MPI_Wait( &blocks[i].req_send_n , status);
        MPI_Wait( &blocks[i].req_recv_n , status);
      }

      // allocate memory
      for (int i = 0; i<n_blocks; i++) blocks[i].allocate_memory();
      
      // receive data
      for (int i = 0; i<n_blocks; i++) blocks[i].send_data();

      for (int i = 0; i<n_blocks; i++) MPI_Waitall( 3 , blocks[i].req_send_data, status);
      for (int i = 0; i<n_blocks; i++) MPI_Waitall( 3 , blocks[i].req_recv_data, status);

      // for (int i = 0; i<n_blocks; i++)
      // printf("Block = %d, rank = %d, received: %d %d %d %d | %d %d %d %d | %f %f %f %f\n",
            // i, rank, blocks[i].orow[0], blocks[i].orow[1], blocks[i].orow[2], blocks[i].orow[3], 
            //   blocks[i].ocol[0], blocks[i].ocol[1], blocks[i].ocol[2], blocks[i].ocol[3],
            //   blocks[i].oval[0], blocks[i].oval[1], blocks[i].oval[2], blocks[i].oval[3]);

  }

  void Forward::copy(int64 *indices, double *val){
    int k = 0;
    for(int i = 0; i<n_blocks; i++){
      auto block = blocks[i];
      for (int j = 0; j<block.recv_n; j++){
        // printf("b%d, %d, --> (%d %d %f)\n", i, j, block.orow[j], block.get_physical_col(j), block.oval[j]);
        indices[2*k] = block.orow[j];
        indices[2*k+1] = block.get_physical_col(j);
        val[k] = block.oval[j];
        k++;
      }
    }
  }
}

