#pragma once

#ifdef BAH_NO_C_API
#define BAH_CBLAS_API
#else
#define BAH_CBLAS_API extern "C"
#endif

namespace bah {

/**
cblas_?asum

The ?asum routine computes the sum of the magnitudes of elements of a real
vector, or the sum of magnitudes of the real and imaginary parts of elements of
a complex vector:

res = |Re x1| + |Im x1| + |Re x2| + Im x2|+ ... + |Re xn| + |Im xn|,

Input Parameters
  n: Specifies the number of elements in vector x.
  x: Array, size at least (1 + (n-1)*abs(incx)).
  incx: Specifies the increment for indexing vector x.

Output Parameters
res: Contains the sum of magnitudes of real and imaginary parts of all elements
of the vector.

Return Values
Contains the sum of magnitudes of real and imaginary parts of all elements of
the vector.

https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/cblas-asum.html
*/
BAH_CBLAS_API float cblas_sasum(const int n, const float *x, const int incX);
BAH_CBLAS_API float cblas_scasum(const int n, const void *x, const int incx);
BAH_CBLAS_API double cblas_dasum(const int n, const double *x, const int incx);
BAH_CBLAS_API double cblas_dzasum(const int n, const void *x, const int incx);

/**
cblas_?axpy

Computes a vector-scalar product and adds the result to a vector
The ?axpy routines perform a vector-vector operation defined as

y := a*x + y

Input Params
  n: Specifies the number of elements in vectors x and y.
  a: Specifies the scalar a.
  x: Array, size at least (1 + (n-1)*abs(incx)).
  incx: Specifies the increment for the elements of x.
  y: Array, size at least (1 + (n-1)*abs(incy)).
  incy: Specifies the increment for the elements of y.

Output Parameters
y: Contains the updated vector y.

https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/cblas-axpy.html
*/
BAH_CBLAS_API void cblas_saxpy(const int n, const float a, const float *x,
                               const int incx, float *y, const int incy);
BAH_CBLAS_API void cblas_daxpy(const int n, const double a, const double *x,
                               const int incx, double *y, const int incy);
BAH_CBLAS_API void cblas_caxpy(const int n, const void *a, const void *x,
                               const int incx, void *y, const int incy);
BAH_CBLAS_API void cblas_zaxpy(const int n, const void *a, const void *x,
                               const int incx, void *y, const int incy);

/**
cblas_?copy

Copies a vector to another vector.

Input Parameters
n: Specifies the number of elements in vectors x and y.
x: Array, size at least (1 + (n-1)*abs(incx)).
incx: Specifies the increment for the elements of x.
y: Array, size at least (1 + (n-1)*abs(incy)).
incy: Specifies the increment for the elements of y.

Output Parameters
y: Contains a copy of the vector x if n is positive. Otherwise, parameters are
unaltered.
*/
BAH_CBLAS_API void cblas_scopy(const int n, const float *x, const int incx,
                               float *y, const int incy);
BAH_CBLAS_API void cblas_dcopy(const int n, const double *x, const int incx,
                               double *y, const int incy);
BAH_CBLAS_API void cblas_ccopy(const int n, const void *x, const int incx,
                               void *y, const int incy);
BAH_CBLAS_API void cblas_zcopy(const int n, const void *x, const int incx,
                               void *y, const int incy);

/**
cblas_?dot

Computes a vector-vector dot product.

Input Parameters
n: Specifies the number of elements in vectors x and y.
x: Array, size at least (1+(n-1)*abs(incx)).
incx: Specifies the increment for the elements of x.
y: Array, size at least (1+(n-1)*abs(incy)).
incy: Specifies the increment for the elements of y.

Return Values
The result of the dot product of x and y, if n is positive. Otherwise, returns
0.
*/
BAH_CBLAS_API float cblas_sdot(const int n, const float *x, const int incx,
                               const float *y, const int incy);
BAH_CBLAS_API double cblas_ddot(const int n, const double *x, const int incx,
                                const double *y, const int incy);

/**
cblas_?sdot

Computes a vector-vector dot product with double precision.

The ?sdot routines compute the inner product of two vectors with double
precision. Both routines use double precision accumulation of the intermediate
results, but the sdsdot routine outputs the final result in single precision,
whereas the dsdot routine outputs the double precision result. The function
sdsdot also adds scalar value sb to the inner product.

Input Parameters
n: Specifies the number of elements in the input vectors sx and sy.
sb: Single precision scalar to be added to inner product (for the function
sdsdot only). sx, sy: Arrays, size at least (1+(n -1)*abs(incx)) and
(1+(n-1)*abs(incy)), respectively. Contain the input single precision vectors.
incx: Specifies the increment for the elements of sx.
incy: Specifies the increment for the elements of sy.

Output Parameters
res: Contains the result of the dot product of sx and sy (with sb added for
sdsdot), if n is positive. Otherwise, res contains sb for sdsdot and 0 for
dsdot.

Return Values
The result of the dot product of sx and sy (with sb added for sdsdot), if n is
positive. Otherwise, returns sb for sdsdot and 0 for dsdot.
*/
float cblas_sdsdot(const int n, const float sb, const float *sx, const int incx,
                   const float *sy, const int incy);
double cblas_dsdot(const int n, const float *sx, const int incx,
                   const float *sy, const int incy);
}  // namespace bah
