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
BAH_CBLAS_API float cblas_sdsdot(const int n, const float sb, const float *sx,
                                 const int incx, const float *sy,
                                 const int incy);
BAH_CBLAS_API double cblas_dsdot(const int n, const float *sx, const int incx,
                                 const float *sy, const int incy);

/**
cblas_?dotc

Computes a dot product of a conjugated vector with another vector.

The ?dotc routines perform a vector-vector operation defined as:

res = \sum_{i=1}^{n} conjg(x_i) * y_i;

where x_i and y_i are elements of vectors x and y.

Input Parameters
n: Specifies the number of elements in vectors x and y.
x: Array, size at least (1 + (n -1)*abs(incx)).
incx: Specifies the increment for the elements of x.
y: Array, size at least (1 + (n -1)*abs(incy)).
incy: Specifies the increment for the elements of y.
Output Parameters
dotc: Contains the result of the dot product of the conjugated x and
unconjugated y, if n is positive. Otherwise, it contains 0.
*/
BAH_CBLAS_API void cblas_cdotc_sub(const int n, const void *x, const int incx,
                                   const void *y, const int incy, void *dotc);
BAH_CBLAS_API void cblas_zdotc_sub(const int n, const void *x, const int incx,
                                   const void *y, const int incy, void *dotc);

/**
cblas_?dotu

Computes a complex vector-vector dot product.

The ?dotu routines perform a vector-vector reduction operation defined as

res = \sum_{i=1}^{n} x_i * y_i;

where xi and yi are elements of complex vectors x and y.

Input Parameters
n: Specifies the number of elements in vectors x and y.
x: Array, size at least (1 + (n -1)*abs(incx)).
incx: Specifies the increment for the elements of x.
y: Array, size at least (1 + (n -1)*abs(incy)).
incy: Specifies the increment for the elements of y.

Output Parameters
dotu: Contains the result of the dot product of x and y, if n is positive.
Otherwise, it contains 0.
*/
BAH_CBLAS_API void cblas_cdotu_sub(const int n, const void *x, const int incx,
                                   const void *y, const int incy, void *dotu);
BAH_CBLAS_API void cblas_zdotu_sub(const int n, const void *x, const int incx,
                                   const void *y, const int incy, void *dotu);

/**
cblas_?nrm2

Computes the Euclidean norm of a vector.

The ?nrm2 routines perform a vector reduction operation defined as

res = ||x||

where x is a vector, res is a value containing the Euclidean norm of the
elements of x.

Input Parameters
n: Specifies the number of elements in vector x.
x: Array, size at least (1 + (n -1)*abs (incx)).
incx: Specifies the increment for the elements of x.

Return Values
The Euclidean norm of the vector x.
*/
BAH_CBLAS_API float cblas_snrm2(const int n, const float *x, const int incx);
BAH_CBLAS_API double cblas_dnrm2(const int n, const double *x, const int incx);
BAH_CBLAS_API float cblas_scnrm2(const int n, const void *x, const int incx);
BAH_CBLAS_API double cblas_dznrm2(const int n, const void *x, const int incx);

/**
cblas_?rot

Performs rotation of points in the plane.

Given two complex vectors x and y, each vector element of these vectors is
replaced as follows:

  xi = c*xi + s*yi
  yi = c*yi - s*xi

If s is a complex type, each vector element is replaced as follows:

  xi = c*xi + s*yi
  yi = c*yi - conj(s)*xi

Input Parameters
n: Specifies the number of elements in vectors x and y.
x: Array, size at least (1 + (n-1)*abs(incx)).
incx: Specifies the increment for the elements of x.
y: Array, size at least (1 + (n -1)*abs(incy)).
incy: Specifies the increment for the elements of y.
c: A scalar.
s: A scalar.

Output Parameters
x: Each element is replaced by c*x + s*y.
y: Each element is replaced by c*y - s*x, or by c*y-conj(s)*x if s is a complex
type.
*/
BAH_CBLAS_API void cblas_srot(const int n, float *x, const int incx, float *y,
                              const int incy, const float c, const float s);
BAH_CBLAS_API void cblas_drot(const int n, double *x, const int incx, double *y,
                              const int incy, const double c, const double s);
BAH_CBLAS_API void cblas_crot(const int n, void *x, const int incx, void *y,
                              const int incy, const float c, const void *s);
BAH_CBLAS_API void cblas_zrot(const int n, void *x, const int incx, void *y,
                              const int incy, const double c, const void *s);
BAH_CBLAS_API void cblas_csrot(const int n, void *x, const int incx, void *y,
                               const int incy, const float c, const float s);
BAH_CBLAS_API void cblas_zdrot(const int n, void *x, const int incx, void *y,
                               const int incy, const double c, const double s);

}  // namespace bah
