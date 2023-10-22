#include "bah_level1.hpp"

#include <cmath>

// BLAS at home
namespace bah {

// cblas_?asum
template <typename RealTp>
static inline RealTp asum_kernel(const int n, const RealTp *x, const int incx) {
  RealTp sum{0.0};
  for (int i = 0; i < n; i += incx) {
    sum += std::fabs(x[i]);
  }
  return sum;
}

template <typename RealTp>
static inline RealTp casum_kernel(const int n, const RealTp *x,
                                  const int incx) {
  RealTp sum{0.0};
  for (int i = 0; i < n * 2; i += incx * 2) {
    // Real and Imag
    sum += std::fabs(x[i]) + std::fabs(x[i + 1]);
  }
  return sum;
}

float cblas_sasum(const int n, const float *x, const int incx) {
  return asum_kernel<float>(n, x, incx);
}

double cblas_dasum(const int n, const double *x, const int incx) {
  return asum_kernel<double>(n, x, incx);
}

float cblas_scasum(const int n, const void *x, const int incx) {
  return casum_kernel<float>(n, reinterpret_cast<const float *>(x), incx);
}

double cblas_dzasum(const int n, const void *x, const int incx) {
  return casum_kernel<double>(n, reinterpret_cast<const double *>(x), incx);
}

// cblas_?axpy
template <typename RealTp>
static inline void axpy_kernel(const int n, const RealTp a, const RealTp *x,
                               const int incx, RealTp *y, const int incy) {
  int ix = 0, iy = 0;
  for (int i = 0; i < n; i++) {
    y[iy] += a * x[ix];
    ix += incx;
    iy += incy;
  }
}

void cblas_saxpy(const int n, const float a, const float *x, const int incx,
                 float *y, const int incy) {
  return axpy_kernel<float>(n, a, x, incx, y, incy);
}
void cblas_daxpy(const int n, const double a, const double *x, const int incx,
                 double *y, const int incy) {
  return axpy_kernel<double>(n, a, x, incx, y, incy);
}
void cblas_caxpy(const int n, const void *a, const void *x, const int incx,
                 void *y, const int incy) {
  // TODO
  float _a = *(reinterpret_cast<const float *>(a));
  // return axpy_kernel<float>(n, , x, incx*2, reinterpret_cast<const
  // float*>(y), incy*2);
};
void cblas_zaxpy(const int n, const void *a, const void *x, const int incx,
                 void *y, const int incy){
    // TODO
};

}  // namespace bah
