#ifndef AA_BLAS_H_GUARD
#define AA_BLAS_H_GUARD

#ifdef __cplusplus
extern "C" {
#endif

#include "aa.h"

/* Default to underscore for blas / lapack */
#ifndef BLASSUFFIX
#define BLASSUFFIX _
#endif

/* annoying hack because some preprocessors can't handle empty macros */
#if defined(NOBLASSUFFIX) && NOBLASSUFFIX > 0
/* single or double precision */
#ifndef SFLOAT
#define BLAS(x) d##x
#else
#define BLAS(x) s##x
#endif
#else
/* this extra indirection is needed for BLASSUFFIX to work correctly as a
 * variable */
#define stitch_(pre, x, post) pre##x##post
#define stitch__(pre, x, post) stitch_(pre, x, post)
/* single or double precision */
#ifndef SFLOAT
#define BLAS(x) stitch__(d, x, BLASSUFFIX)
#else
#define BLAS(x) stitch__(s, x, BLASSUFFIX)
#endif
#endif

#ifdef MATLAB_MEX_FILE
typedef ptrdiff_t blas_int;
#elif defined BLAS64
#include <stdint.h>
typedef int64_t blas_int;
#else
typedef int blas_int;
#endif

/* BLAS functions used */
aa_float BLAS(nrm2)(blas_int *n, aa_float *x, blas_int *incx);
void BLAS(axpy)(blas_int *n, aa_float *a, const aa_float *x, blas_int *incx,
                aa_float *y, blas_int *incy);
void BLAS(gemv)(const char *trans, const blas_int *m, const blas_int *n,
                const aa_float *alpha, const aa_float *a, const blas_int *lda,
                const aa_float *x, const blas_int *incx, const aa_float *beta,
                aa_float *y, const blas_int *incy);
void BLAS(gesv)(blas_int *n, blas_int *nrhs, aa_float *a, blas_int *lda,
                blas_int *ipiv, aa_float *b, blas_int *ldb, blas_int *info);
void BLAS(gemm)(const char *transa, const char *transb, blas_int *m,
                blas_int *n, blas_int *k, aa_float *alpha, aa_float *a,
                blas_int *lda, aa_float *b, blas_int *ldb, aa_float *beta,
                aa_float *c, blas_int *ldc);

#ifdef __cplusplus
}
#endif

#endif /* AA_BLAS_H_GUARD */
