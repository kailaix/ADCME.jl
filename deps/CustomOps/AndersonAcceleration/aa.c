#include "aa.h"

#include "aa_blas.h"

#if PROFILING > 0

#define TIME_TIC \
  timer __t;     \
  tic(&__t);
#define TIME_TOC toc(__func__, &__t);

#include <time.h>
typedef struct timer {
  struct timespec tic;
  struct timespec toc;
} timer;

void tic(timer *t) { clock_gettime(CLOCK_MONOTONIC, &t->tic); }

aa_float tocq(timer *t) {
  struct timespec temp;

  clock_gettime(CLOCK_MONOTONIC, &t->toc);

  if ((t->toc.tv_nsec - t->tic.tv_nsec) < 0) {
    temp.tv_sec = t->toc.tv_sec - t->tic.tv_sec - 1;
    temp.tv_nsec = 1e9 + t->toc.tv_nsec - t->tic.tv_nsec;
  } else {
    temp.tv_sec = t->toc.tv_sec - t->tic.tv_sec;
    temp.tv_nsec = t->toc.tv_nsec - t->tic.tv_nsec;
  }
  return (aa_float)temp.tv_sec * 1e3 + (aa_float)temp.tv_nsec / 1e6;
}

aa_float toc(const char *str, timer *t) {
  aa_float time = tocq(t);
  printf("%s - time: %8.4f milli-seconds.\n", str, time);
  return time;
}

#else

#define TIME_TIC
#define TIME_TOC

#endif

/* This file uses Anderson acceleration to improve the convergence of
 * a fixed point mapping.
 * At each iteration we need to solve a (small) linear system, we
 * do this using LAPACK ?gesv.
 */

/* contains the necessary parameters to perform aa at each step */
struct ACCEL_WORK {
  aa_int type1; /* bool, if true type 1 aa otherwise type 2 */
  aa_int mem;   /* aa memory */
  aa_int dim;   /* variable dimension */
  aa_int iter;  /* current iteration */

  aa_float *x; /* x input to map*/
  aa_float *f; /* f(x) output of map */
  aa_float *g; /* x - f(x) */

  /* from previous iteration */
  aa_float *g_prev; /* x - f(x) */

  aa_float *y; /* g - g_prev */
  aa_float *s; /* x - x_prev */
  aa_float *d; /* f - f_prev */

  aa_float *Y; /* matrix of stacked y values */
  aa_float *S; /* matrix of stacked s values */
  aa_float *D; /* matrix of stacked d values = (S-Y) */
  aa_float *M; /* S'Y or Y'Y depending on type of aa */

  /* workspace variables */
  aa_float *work;
  blas_int *ipiv;
};

/* sets a->M to S'Y or Y'Y depending on type of aa used */
static void set_m(AaWork *a) {
  TIME_TIC
  blas_int bdim = (blas_int)(a->dim), bmem = (blas_int)a->mem;
  aa_float onef = 1.0, zerof = 0.0;
  BLAS(gemm)
  ("Trans", "No", &bmem, &bmem, &bdim, &onef, a->type1 ? a->S : a->Y, &bdim,
   a->Y, &bdim, &zerof, a->M, &bmem);
  TIME_TOC
  return;
}

/* updates the workspace parameters for aa for this iteration */
static void update_accel_params(const aa_float *x, const aa_float *f,
                                AaWork *a) {
  /* at the start a->x = x_prev and a->f = f_prev */
  TIME_TIC
  aa_int idx = a->iter % a->mem;
  blas_int one = 1;
  blas_int bdim = (blas_int)a->dim;
  aa_float neg_onef = -1.0;

  /* g = x */
  memcpy(a->g, x, sizeof(aa_float) * a->dim);
  /* s = x */
  memcpy(a->s, x, sizeof(aa_float) * a->dim);
  /* d = f */
  memcpy(a->d, f, sizeof(aa_float) * a->dim);
  /* g -= f */
  BLAS(axpy)(&bdim, &neg_onef, f, &one, a->g, &one);
  /* s -= x_prev */
  BLAS(axpy)(&bdim, &neg_onef, a->x, &one, a->s, &one);
  /* d -= f_prev */
  BLAS(axpy)(&bdim, &neg_onef, a->f, &one, a->d, &one);

  /* g, s, d correct here */

  /* y = g */
  memcpy(a->y, a->g, sizeof(aa_float) * a->dim);
  /* y -= g_prev */
  BLAS(axpy)(&bdim, &neg_onef, a->g_prev, &one, a->y, &one);

  /* y correct here */

  /* copy y into idx col of Y */
  memcpy(&(a->Y[idx * a->dim]), a->y, sizeof(aa_float) * a->dim);
  /* copy s into idx col of S */
  memcpy(&(a->S[idx * a->dim]), a->s, sizeof(aa_float) * a->dim);
  /* copy d into idx col of D */
  memcpy(&(a->D[idx * a->dim]), a->d, sizeof(aa_float) * a->dim);

  /* Y, S, D correct here */

  memcpy(a->f, f, sizeof(aa_float) * a->dim);
  memcpy(a->x, x, sizeof(aa_float) * a->dim);

  /* x, f correct here */

  /* set M = S'Y or Y'Y depending on type of aa used */
  set_m(a);

  /* M correct here */

  memcpy(a->g_prev, a->g, sizeof(aa_float) * a->dim);

  /* g_prev set for next iter here */

  TIME_TOC
  return;
}

/* solves the system of equations to perform the aa update
 * at the end f contains the next iterate to be returned
 */
static aa_int solve(aa_float *f, AaWork *a, aa_int len) {
  TIME_TIC
  blas_int info = -1, bdim = (blas_int)(a->dim), one = 1, blen = (blas_int)len,
           bmem = (blas_int)a->mem;
  aa_float neg_onef = -1.0, onef = 1.0, zerof = 0.0, nrm;
  /* work = S'g or Y'g */
  BLAS(gemv)
  ("Trans", &bdim, &blen, &onef, a->type1 ? a->S : a->Y, &bdim, a->g, &one,
   &zerof, a->work, &one);
  /* work = M \ work, where M = S'Y or M = Y'Y */
  BLAS(gesv)(&blen, &one, a->M, &bmem, a->ipiv, a->work, &blen, &info);
  nrm = BLAS(nrm2)(&bmem, a->work, &one);
  if (info < 0 || nrm >= MAX_AA_NRM) {
    /* printf("Error in AA type %i, iter: %i, info: %i, norm %.2e\n", */
    /*       a->type1 ? 1 : 2, (int)a->iter, (int)info, nrm);         */
    return -1;
  }
  /* if solve was successful then set f -= D * work */
  BLAS(gemv)
  ("NoTrans", &bdim, &blen, &neg_onef, a->D, &bdim, a->work, &one, &onef, f,
   &one);
  TIME_TOC
  return (aa_int)info;
}

/*
 * API functions below this line, see aa.h for descriptions.
 */
AaWork *aa_init(aa_int dim, aa_int mem, aa_int type1) {
  AaWork *a = (AaWork *)calloc(1, sizeof(AaWork));
  if (!a) {
    printf("Failed to allocate memory for AA.\n");
    return (void *)0;
  }
  a->type1 = type1;
  a->iter = 0;
  a->dim = dim;
  a->mem = mem;
  if (a->mem <= 0) {
    return a;
  }

  a->x = (aa_float *)calloc(a->dim, sizeof(aa_float));
  a->f = (aa_float *)calloc(a->dim, sizeof(aa_float));
  a->g = (aa_float *)calloc(a->dim, sizeof(aa_float));

  a->g_prev = (aa_float *)calloc(a->dim, sizeof(aa_float));

  a->y = (aa_float *)calloc(a->dim, sizeof(aa_float));
  a->s = (aa_float *)calloc(a->dim, sizeof(aa_float));
  a->d = (aa_float *)calloc(a->dim, sizeof(aa_float));

  a->Y = (aa_float *)calloc(a->dim * a->mem, sizeof(aa_float));
  a->S = (aa_float *)calloc(a->dim * a->mem, sizeof(aa_float));
  a->D = (aa_float *)calloc(a->dim * a->mem, sizeof(aa_float));

  a->M = (aa_float *)calloc(a->mem * a->mem, sizeof(aa_float));
  a->work = (aa_float *)calloc(a->mem, sizeof(aa_float));
  a->ipiv = (blas_int *)calloc(a->mem, sizeof(blas_int));
  return a;
}

aa_int aa_apply(aa_float *f, const aa_float *x, AaWork *a) {
  TIME_TIC
  aa_int status;
  if (a->mem <= 0) {
    return 0;
  }
  update_accel_params(x, f, a);
  if (a->iter++ == 0) {
    return 0;
  }
  /* solve linear system, new point overwrites f if successful */
  status = solve(f, a, MIN(a->iter - 1, a->mem));
  TIME_TOC
  return status;
}

void aa_finish(AaWork *a) {
  if (a) {
    free(a->x);
    free(a->f);
    free(a->g);
    free(a->g_prev);
    free(a->y);
    free(a->s);
    free(a->d);
    free(a->Y);
    free(a->S);
    free(a->D);
    free(a->M);
    free(a->work);
    free(a->ipiv);
    free(a);
  }
  return;
}
