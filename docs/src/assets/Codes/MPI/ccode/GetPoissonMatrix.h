namespace PoissonMPI{

   #include <stdio.h>
   #include <stdlib.h>
   #include <string.h>
   #include <math.h>
   #include "HYPRE_parcsr_ls.h"
   #include "HYPRE.h"
   #include <algorithm>

   #define kext(i,j) kext[(i)*(n+2) + (j)]
   #define uext(i,j) uext[(i)*(n+2) + (j)]
   #define colext(i,j) colext[(i)*(n+2) + (j)]

   void GetPoissonMatrixForward(
     int64 *cols,
     double *val,
     const int64  *colext, 
     const double *kext, int n){

       for(int i = 1; i < n+1; i++){
         for(int j = 1; j < n+1; j++){

           if (colext(i+1, j)>=0){
             *(cols++) = colext(i+1, j);
             *(val++) = kext(i+1, j) + kext(i, j);
           }

           if (colext(i-1, j)>=0){
             *(cols++) = colext(i-1, j);
             *(val++) = kext(i-1, j) + kext(i, j);
           }

           if (colext(i, j+1)>=0){
             *(cols++) = colext(i, j+1);
             *(val++) = kext(i, j) + kext(i, j+1);
           }

           if (colext(i, j-1)>=0){
             *(cols++) = colext(i, j-1);
             *(val++) = kext(i, j) + kext(i, j-1);
           }

           *(cols++) = colext(i, j);
           *(val++) = -(kext(i, j+1) + kext(i, j-1) + kext(i+1, j) + kext(i-1, j) + 4 * kext(i,j));

         }


       }
     }

      
       #define xgrad(i,j) xgrad[((i)-1)*n + ((j)-1)]
       void GetPoissonGrad(
         double *kext, 
        const double *xgrad, const double *uext, int N, int n){

         memset(kext, 0, sizeof(double)*(n+2)*(n+2));
         for(int i = 1; i < n+1; i++){
           for(int j = 1; j < n+1; j++){
             kext(i+1, j) += xgrad(i, j) * (uext(i+1, j) - uext(i, j));
             kext(i-1, j) += xgrad(i, j) * (uext(i-1, j) - uext(i, j));
             kext(i, j+1) += xgrad(i, j) * (uext(i, j+1) - uext(i, j));
             kext(i, j-1) += xgrad(i, j) * (uext(i, j-1) - uext(i, j));
             kext(i, j) += xgrad(i, j) * (-4*uext(i, j) + uext(i+1, j) + \
                                  uext(i-1, j) + uext(i, j+1) + uext(i, j-1));
            // printf("xgrad(%d,%d)=%f, uext = %f\n", i, j, xgrad(i,j), uext(i,j));
           }
         }

       }
    
  
}