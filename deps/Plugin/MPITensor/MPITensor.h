int MPICreateMatrix_GetNumberOfRows(const int64 *indices, int n){
    if (n==0) return 0;
    int64 cur = indices[0];
    int nrow = 1;
    for (int i = 1; i<n; i++){
        if (indices[2*i]==cur) continue;
        else {
            nrow++;
            cur = indices[2*i];
        }
    }
    return nrow;
}

void MPICreateMatrix_forward(
    int *rows, int *ncols, int *cols,
    const int64 *indices, int n, int64 ilower, int64 iupper
){
    if (n==0) return;
    int64 cur = indices[0];
    rows[0] = (int)cur + ilower;
    ncols[0] = 1;
    cols[0] = (int)indices[1];
    int nrow = 1;
    for (int i = 1; i<n; i++){
        if (indices[2*i]==cur) {
            ncols[nrow-1]++;
        }
        else {
            rows[nrow++] = (int)indices[2*i] + ilower;
            ncols[nrow-1] = 1;
            cur = indices[2*i];
        }
        cols[i] = indices[2*i+1];
    }
}


void MPIGetMatrix_forward(
    int64 *indices, double *vv,
    const int *rows, const int *ncols, const int *cols,const double *values,
    int n, int nrows, int64 ilower, int64 iupper){
    for(int i = 0; i<n; i++){
        vv[i] = values[i];
        indices[2*i+1] = cols[i];
    }
    int nz = 0;
    for(int k = 0; k < nrows; k++){
        for(int j = 0; j<ncols[k]; j++) indices[2*(nz++)] = rows[k]-ilower;
    }
}


void mpi_solve_BoomerAMG(HYPRE_IJMatrix &A, HYPRE_IJVector &x, HYPRE_IJVector &b, int PrintLevel){
   HYPRE_ParCSRMatrix    par_A;
   HYPRE_ParVector       par_b;
   HYPRE_ParVector       par_x;
   HYPRE_IJMatrixGetObject(A, (void **)& par_A);
   HYPRE_IJVectorGetObject(b, (void **)& par_b);
   HYPRE_IJVectorGetObject(x, (void **)& par_x);
   
   HYPRE_Solver solver;
   HYPRE_BoomerAMGCreate(&solver);
   HYPRE_BoomerAMGSetOldDefault(solver);
   HYPRE_BoomerAMGSetStrongThreshold(solver, 0.25);
   HYPRE_BoomerAMGSetTol(solver, 1e-10);
   HYPRE_BoomerAMGSetPrintLevel(solver, PrintLevel);
   HYPRE_BoomerAMGSetMaxIter(solver, 200);

   /* call the setup */
   HYPRE_BoomerAMGSetup(solver, par_A, par_b, par_x);

   /* call the solve */
   HYPRE_BoomerAMGSolve(solver, par_A, par_b, par_x);
   HYPRE_BoomerAMGDestroy(solver);
}


void mpi_solve_GMRES(HYPRE_IJMatrix &A, HYPRE_IJVector &x, HYPRE_IJVector &b, int PrintLevel){
   HYPRE_ParCSRMatrix    par_A;
   HYPRE_ParVector       par_b;
   HYPRE_ParVector       par_x;
   HYPRE_IJMatrixGetObject(A, (void **)& par_A);
   HYPRE_IJVectorGetObject(b, (void **)& par_b);
   HYPRE_IJVectorGetObject(x, (void **)& par_x);

   MPI_Comm comm = MPI_COMM_WORLD;
   HYPRE_Solver solver;
   HYPRE_ParCSRGMRESCreate(comm, &solver);
   HYPRE_ParCSRGMRESSetTol(solver, 1e-10);
   HYPRE_ParCSRGMRESSetPrintLevel(solver, PrintLevel);
   HYPRE_ParCSRGMRESSetMaxIter(solver, 200);

   /* call the setup */
   HYPRE_ParCSRGMRESSetup(solver, par_A, par_b, par_x);

   /* call the solve */
   HYPRE_ParCSRGMRESSolve(solver, par_A, par_b, par_x);
   HYPRE_ParCSRGMRESDestroy(solver);
}


void MPITensorSolve_forward(
        double *out,
        const int32* rows, const int32* ncols, const int32* cols, const double* values, 
        const double *rhs, 
        int nrows, int nnz,
        int ilower, int iupper, int PrintLevel, string Solver){

   MPI_Comm comm = MPI_COMM_WORLD;
   // solve the linear system
   HYPRE_IJVector   b;
   HYPRE_IJVector   x;
   HYPRE_IJMatrix A;
   HYPRE_IJVectorCreate(comm, ilower, iupper, &b);
   HYPRE_IJVectorCreate(comm, ilower, iupper, &x);
   HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &A);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(x);
   HYPRE_IJVectorInitialize(b);
   HYPRE_IJMatrixInitialize(A);
   double zero = 0.0;
   for(int i = ilower; i<=iupper;i++){
      HYPRE_IJVectorSetValues(b, 1, &i, rhs + i - ilower);
      HYPRE_IJVectorSetValues(x, 1, &i, &zero);
   }
   auto ncols_ = new int[nrows];
   for(int i = 0; i<nrows; i++) ncols_[i] = ncols[i];
   HYPRE_IJMatrixSetValues(A, nrows, ncols_, rows, cols, values);
   
   HYPRE_IJVectorAssemble(b);
   HYPRE_IJVectorAssemble(x);
   HYPRE_IJMatrixAssemble(A);

   if(Solver.compare("BoomerAMG")==0)
    mpi_solve_BoomerAMG(A, x, b, PrintLevel);
   else if(Solver.compare("GMRES")==0)
    mpi_solve_GMRES(A, x, b, PrintLevel);
   else 
    throw;


   for (int i = ilower; i <= iupper; i++)
      HYPRE_IJVectorGetValues(x, 1, &i, out+i-ilower);  
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJMatrixDestroy(A);

   delete [] ncols_;
}

