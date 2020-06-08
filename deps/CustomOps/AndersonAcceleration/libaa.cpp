#ifdef _WIN32
#define EXPORTED  __declspec( dllexport )
#else
#define EXPORTED
#endif 

extern "C" {
#include "aa.h"
#include "aa_blas.h"
}
AaWork * aa_global;

extern "C" EXPORTED void init_aa(int dim, int mem, int type1){
    aa_global = aa_init(dim, mem, type1);
};


extern "C" EXPORTED void apply_aa(double *f, const double *x){
    aa_apply(f, x, aa_global);
};

extern "C" EXPORTED void finalize_aa(){
    aa_finish(aa_global);
};