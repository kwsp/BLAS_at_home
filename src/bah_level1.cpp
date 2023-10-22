#include "bah_level1.hpp"

#include <cmath>
#include <complex>

// BLAS at home
namespace bah {

namespace {
/**
Multiple 2 complex numbers (a, b) and add the result to res.
*/
template <typename T>
static inline void cx_mul(const T *a, const T *b, T *res) {
  // This doesn't handle the infinity edge cases but is much faster than the std
  // version.
  const T &re1 = a[0], re2 = b[0];
  const T &im1 = a[1], im2 = b[1];
  res[0] += re1 * re2 - im1 * im2;
  res[1] += re1 * im2 + re2 * im1;

  // const auto &_a = *reinterpret_cast<const std::complex<T> *>(a);
  // const auto &_b = *reinterpret_cast<const std::complex<T> *>(b);
  // const auto _res = _a * _b;
  // res[0] += _res.real();
  // res[1] += _res.imag();
}

/**
Multiple 2 complex numbers (conjg(a), b) and add the result to res.
*/
template <typename T>
static inline void cx_mulc(const T *a, const T *b, T *res) {
  // This doesn't handle the infinity edge cases but is much faster than the std
  // version.
  const T &re1 = a[0], re2 = b[0];
  const T &im1 = a[1], im2 = b[1];
  res[0] += re1 * re2 + im1 * im2;
  res[1] += re1 * im2 - re2 * im1;

  // const auto &_a = *reinterpret_cast<const std::complex<T> *>(a);
  // const auto &_b = *reinterpret_cast<const std::complex<T> *>(b);
  // const auto _res = std::conj(_a) * _b;
  // res[0] += _res.real();
  // res[1] += _res.imag();
}

}  // namespace

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
  int ix{0}, iy{0};
  for (int i = 0; i < n; i++) {
    y[iy] += a * x[ix];

    ix += incx;
    iy += incy;
  }
}

template <typename RealTp>
static inline void caxpy_kernel(const int n, const RealTp *a, const RealTp *x,
                                const int incx, RealTp *y, const int incy) {
  int ix{0}, iy{0};
  for (int i = 0; i < n; i++) {
    cx_mul(a, x + ix, y + iy);
    ix += incx * 2;
    iy += incy * 2;
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
  return caxpy_kernel<float>(n, reinterpret_cast<const float *>(a),
                             reinterpret_cast<const float *>(x), incx,
                             reinterpret_cast<float *>(y), incy);
};
void cblas_zaxpy(const int n, const void *a, const void *x, const int incx,
                 void *y, const int incy) {
  return caxpy_kernel<double>(n, reinterpret_cast<const double *>(a),
                              reinterpret_cast<const double *>(x), incx,
                              reinterpret_cast<double *>(y), incy);
};

// cblas_?copy
template <typename RealTp>
static inline void _copy_kernel(const int n, const RealTp *x, const int incx,
                                RealTp *y, const int incy) {
  int ix{0}, iy{0};
  for (int i = 0; i < n; i++) {
    y[iy] = x[ix];

    ix += incx;
    iy += incy;
  }
}

template <typename RealTp>
static inline void _ccopy_kernel(const int n, const RealTp *x, const int incx,
                                 RealTp *y, const int incy) {
  int ix{0}, iy{0};
  for (int i = 0; i < n; i++) {
    y[iy] = x[ix];
    y[iy + 1] = x[ix + 1];

    ix += incx * 2;
    iy += incy * 2;
  }
}

void cblas_scopy(const int n, const float *x, const int incx, float *y,
                 const int incy) {
  _copy_kernel<float>(n, x, incx, y, incy);
}

void cblas_dcopy(const int n, const double *x, const int incx, double *y,
                 const int incy) {
  _copy_kernel<double>(n, x, incx, y, incy);
}

void cblas_ccopy(const int n, const void *x, const int incx, void *y,
                 const int incy) {
  _ccopy_kernel<float>(n, reinterpret_cast<const float *>(x), incx,
                       reinterpret_cast<float *>(y), incy);
}

void cblas_zcopy(const int n, const void *x, const int incx, void *y,
                 const int incy) {
  _ccopy_kernel<double>(n, reinterpret_cast<const double *>(x), incx,
                        reinterpret_cast<double *>(y), incy);
}

// cblas_?dot
template <typename RealTp>
static inline RealTp dot_kernel(const int n, const RealTp *x, const int incx,
                                const RealTp *y, const int incy) {
  RealTp res{};
  int ix{}, iy{};
  for (int i = 0; i < n; i++) {
    res += x[ix] * y[iy];
    ix += incx;
    iy += incy;
  }
  return res;
}

float cblas_sdot(const int n, const float *x, const int incx, const float *y,
                 const int incy) {
  return dot_kernel(n, x, incx, y, incy);
}
double cblas_ddot(const int n, const double *x, const int incx, const double *y,
                  const int incy) {
  return dot_kernel(n, x, incx, y, incy);
}

// cblas_?sdot
static inline double dsdot_kernel(const int n, const float *x, const int incx,
                                  const float *y, const int incy) {
  double res{};
  int ix{}, iy{};
  for (int i = 0; i < n; i++) {
    res += static_cast<double>(x[ix]) * static_cast<double>(y[iy]);
    ix += incx;
    iy += incy;
  }
  return res;
}

float cblas_sdsdot(const int n, const float sb, const float *sx, const int incx,
                   const float *sy, const int incy) {
  const double res = dsdot_kernel(n, sx, incx, sy, incy);
  return static_cast<float>(res) + sb;
}

double cblas_dsdot(const int n, const float *sx, const int incx,
                   const float *sy, const int incy) {
  return dsdot_kernel(n, sx, incx, sy, incy);
}

// cblas_?dotc
template <typename RealTp>
static inline void dotc_kernel(const int n, const RealTp *x, const int incx,
                               const RealTp *y, const int incy, RealTp *dotc) {
  int ix{0}, iy{0};
  for (int i = 0; i < n; i++) {
    cx_mulc<RealTp>(x + ix, y + iy, dotc);
    ix += incx * 2;
    iy += incy * 2;
  }
}

void cblas_cdotc_sub(const int n, const void *x, const int incx, const void *y,
                     const int incy, void *dotc) {
  dotc_kernel(n, reinterpret_cast<const float *>(x), incx,
              reinterpret_cast<const float *>(y), incy,
              reinterpret_cast<float *>(dotc));
}

void cblas_zdotc_sub(const int n, const void *x, const int incx, const void *y,
                     const int incy, void *dotc) {
  dotc_kernel(n, reinterpret_cast<const double *>(x), incx,
              reinterpret_cast<const double *>(y), incy,
              reinterpret_cast<double *>(dotc));
};

// cblas_?dotu
template <typename RealTp>
static inline void dotu_kernel(const int n, const RealTp *x, const int incx,
                               const RealTp *y, const int incy, RealTp *dotc) {
  int ix{0}, iy{0};
  for (int i = 0; i < n; i++) {
    cx_mul<RealTp>(x + ix, y + iy, dotc);
    ix += incx * 2;
    iy += incy * 2;
  }
}

void cblas_cdotu_sub(const int n, const void *x, const int incx, const void *y,
                     const int incy, void *dotu) {
  dotu_kernel(n, reinterpret_cast<const float *>(x), incx,
              reinterpret_cast<const float *>(y), incy,
              reinterpret_cast<float *>(dotu));
}
void cblas_zdotu_sub(const int n, const void *x, const int incx, const void *y,
                     const int incy, void *dotu) {
  dotu_kernel(n, reinterpret_cast<const double *>(x), incx,
              reinterpret_cast<const double *>(y), incy,
              reinterpret_cast<double *>(dotu));
}

}  // namespace bah
