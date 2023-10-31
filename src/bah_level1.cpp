#include "bah_level1.hpp"

#include <cmath>
#include <complex>

#include "bah_concepts.hpp"

// BLAS at home
namespace bah {

namespace {

/**
Multiply a complex number and a Real. Add to res

res += a * b
*/
template <typename T>
static inline void _mul_cx_real(const T *a, const T b, T *res) {
  const T &re1 = a[0], im1 = a[1];
  res[0] += re1 * b;
  res[1] += im1 * b;
}

/**
Multiply a complex number and a Real. Subtract from res

res += a * b
*/
template <typename T>
static inline void _mul_cx_real_sub(const T *a, const T b, T *res) {
  const T &re1 = a[0], im1 = a[1];
  res[0] -= re1 * b;
  res[1] -= im1 * b;
}

/**
Multiply 2 complex numbers (a, b) and add the result to res.

res += a * b
*/
template <typename T>
static inline void _mul_cx(const T *a, const T *b, T *res) {
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
Multiply 2 complex numbers (conjg(a), b) and add the result to res.

res += conj(a) * b
*/
template <typename T>
static inline void _mul_cx_conj(const T *a, const T *b, T *res) {
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

/**
Multiply 2 complex numbers (conjg(a), b) and subtract from the result.

res += conj(a) * b
*/
template <typename T>
static inline void _mul_cx_conj_sub(const T *a, const T *b, T *res) {
  // This doesn't handle the infinity edge cases but is much faster than the std
  // version.
  const T &re1 = a[0], re2 = b[0];
  const T &im1 = a[1], im2 = b[1];
  res[0] -= re1 * re2 + im1 * im2;
  res[1] -= re1 * im2 - re2 * im1;

  // const auto &_a = *reinterpret_cast<const std::complex<T> *>(a);
  // const auto &_b = *reinterpret_cast<const std::complex<T> *>(b);
  // const auto _res = std::conj(_a) * _b;
  // res[0] -= _res.real();
  // res[1] -= _res.imag();
}

}  // namespace

// cblas_?asum
template <concepts::is_real Real>
static inline Real _asum_kernel(const int n, const Real *x, const int incx) {
  Real sum{0.0};
  for (int i = 0; i < n; i += incx) {
    sum += std::fabs(x[i]);
  }
  return sum;
}

template <typename RealTp>
static inline RealTp _casum_kernel(const int n, const RealTp *x,
                                   const int incx) {
  RealTp sum{0.0};
  for (int i = 0; i < n * 2; i += incx * 2) {
    // Real and Imag
    sum += std::fabs(x[i]) + std::fabs(x[i + 1]);
  }
  return sum;
}

float cblas_sasum(const int n, const float *x, const int incx) {
  return _asum_kernel(n, x, incx);
}

double cblas_dasum(const int n, const double *x, const int incx) {
  return _asum_kernel(n, x, incx);
}

float cblas_scasum(const int n, const void *x, const int incx) {
  return _casum_kernel(n, reinterpret_cast<const float *>(x), incx);
}

double cblas_dzasum(const int n, const void *x, const int incx) {
  return _casum_kernel(n, reinterpret_cast<const double *>(x), incx);
}

// cblas_?axpy
template <typename RealTp>
static inline void _axpy_kernel(const int n, const RealTp a, const RealTp *x,
                                const int incx, RealTp *y, const int incy) {
  int ix{0}, iy{0};
  for (int i = 0; i < n; i++) {
    y[iy] += a * x[ix];

    ix += incx;
    iy += incy;
  }
}

template <typename RealTp>
static inline void _caxpy_kernel(const int n, const RealTp *a, const RealTp *x,
                                 const int incx, RealTp *y, const int incy) {
  int ix{0}, iy{0};
  for (int i = 0; i < n; i++) {
    _mul_cx(a, x + ix, y + iy);
    ix += incx * 2;
    iy += incy * 2;
  }
}

void cblas_saxpy(const int n, const float a, const float *x, const int incx,
                 float *y, const int incy) {
  return _axpy_kernel<float>(n, a, x, incx, y, incy);
}
void cblas_daxpy(const int n, const double a, const double *x, const int incx,
                 double *y, const int incy) {
  return _axpy_kernel<double>(n, a, x, incx, y, incy);
}
void cblas_caxpy(const int n, const void *a, const void *x, const int incx,
                 void *y, const int incy) {
  return _caxpy_kernel<float>(n, reinterpret_cast<const float *>(a),
                              reinterpret_cast<const float *>(x), incx,
                              reinterpret_cast<float *>(y), incy);
};
void cblas_zaxpy(const int n, const void *a, const void *x, const int incx,
                 void *y, const int incy) {
  return _caxpy_kernel<double>(n, reinterpret_cast<const double *>(a),
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
static inline RealTp _dot_kernel(const int n, const RealTp *x, const int incx,
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
  return _dot_kernel(n, x, incx, y, incy);
}
double cblas_ddot(const int n, const double *x, const int incx, const double *y,
                  const int incy) {
  return _dot_kernel(n, x, incx, y, incy);
}

// cblas_?sdot
static inline double _dsdot_kernel(const int n, const float *x, const int incx,
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
  const double res = _dsdot_kernel(n, sx, incx, sy, incy);
  return static_cast<float>(res) + sb;
}

double cblas_dsdot(const int n, const float *sx, const int incx,
                   const float *sy, const int incy) {
  return _dsdot_kernel(n, sx, incx, sy, incy);
}

// cblas_?dotc
template <typename RealTp>
static inline void _dotc_kernel(const int n, const RealTp *x, const int incx,
                                const RealTp *y, const int incy, RealTp *dotc) {
  int ix{0}, iy{0};
  for (int i = 0; i < n; i++) {
    _mul_cx_conj(x + ix, y + iy, dotc);
    ix += incx * 2;
    iy += incy * 2;
  }
}

void cblas_cdotc_sub(const int n, const void *x, const int incx, const void *y,
                     const int incy, void *dotc) {
  _dotc_kernel(n, reinterpret_cast<const float *>(x), incx,
               reinterpret_cast<const float *>(y), incy,
               reinterpret_cast<float *>(dotc));
}

void cblas_zdotc_sub(const int n, const void *x, const int incx, const void *y,
                     const int incy, void *dotc) {
  _dotc_kernel(n, reinterpret_cast<const double *>(x), incx,
               reinterpret_cast<const double *>(y), incy,
               reinterpret_cast<double *>(dotc));
};

// cblas_?dotu
template <typename RealTp>
static inline void _dotu_kernel(const int n, const RealTp *x, const int incx,
                                const RealTp *y, const int incy, RealTp *dotc) {
  int ix{0}, iy{0};
  for (int i = 0; i < n; i++) {
    _mul_cx(x + ix, y + iy, dotc);
    ix += incx * 2;
    iy += incy * 2;
  }
}

void cblas_cdotu_sub(const int n, const void *x, const int incx, const void *y,
                     const int incy, void *dotu) {
  _dotu_kernel(n, reinterpret_cast<const float *>(x), incx,
               reinterpret_cast<const float *>(y), incy,
               reinterpret_cast<float *>(dotu));
}
void cblas_zdotu_sub(const int n, const void *x, const int incx, const void *y,
                     const int incy, void *dotu) {
  _dotu_kernel(n, reinterpret_cast<const double *>(x), incx,
               reinterpret_cast<const double *>(y), incy,
               reinterpret_cast<double *>(dotu));
}

// cblas_?nrm2
template <typename RealTp>
static inline RealTp _nrm2_kernel(const int n, const RealTp *x,
                                  const int incx) {
  RealTp sum2{};
  int ix{0};
  for (int i = 0; i < n; i++) {
    sum2 += x[ix] * x[ix];
    ix += incx;
  }
  return std::sqrt(sum2);
}

template <typename RealTp>
static inline RealTp _cnrm2_kernel(const int n, const RealTp *x,
                                   const int incx) {
  RealTp sum2{};
  int ix{0};
  for (int i = 0; i < n; i++) {
    sum2 += x[ix] * x[ix] + x[ix + 1] * x[ix + 1];
    ix += incx * 2;
  }
  return std::sqrt(sum2);
}

float cblas_snrm2(const int n, const float *x, const int incx) {
  return _nrm2_kernel(n, x, incx);
}

double cblas_dnrm2(const int n, const double *x, const int incx) {
  return _nrm2_kernel(n, x, incx);
}

float cblas_scnrm2(const int n, const void *x, const int incx) {
  return _cnrm2_kernel(n, reinterpret_cast<const float *>(x), incx);
}

double cblas_dznrm2(const int n, const void *x, const int incx) {
  return _cnrm2_kernel(n, reinterpret_cast<const double *>(x), incx);
};

// cblas_?rot
template <typename RealTp>
static inline void _rot_kernel(const int n, RealTp *x, const int incx,
                               RealTp *y, const int incy, const RealTp c,
                               const RealTp s) {
  int ix{0}, iy{0};
  for (int i = 0; i < n; i++) {
    RealTp xi = x[ix];
    RealTp yi = y[iy];
    x[ix] = c * xi + s * yi;
    y[iy] = c * yi - s * xi;

    ix += incx;
    iy += incy;
  }
}

template <typename RealTp>
static inline void _cs_rot_kernel(const int n, RealTp *x, const int incx,
                                  RealTp *y, const int incy, const RealTp c,
                                  const RealTp *s) {
  // x and y are complex
  // c is real, s is complex
  int ix{0}, iy{0};
  for (int i = 0; i < n; i++) {
    // xi = c*xi + s*yi
    // TODO x and y are not complex here?
    _mul_cx_real(x + ix, c, x + ix);
    _mul_cx(y + iy, s, x + ix);

    // yi = c*yi - conj(s)*xi
    _mul_cx_real(y + iy, c, y + iy);
    _mul_cx_conj_sub(s, x + ix, y + iy);

    ix += incx * 2;
    iy += incy * 2;
  }
}

template <typename RealTp>
static inline void _crot_kernel(const int n, RealTp *x, const int incx,
                                RealTp *y, const int incy, const RealTp c,
                                const RealTp s) {
  // x and y are complex
  // c and s are real
  int ix{0}, iy{0};
  for (int i = 0; i < n; i++) {
    // xi = c*xi + s*yi
    _mul_cx_real(x + ix, c, x + ix);
    _mul_cx_real(y + iy, s, x + ix);

    // yi = c*yi - conj(s)*xi
    _mul_cx_real(y + iy, c, y + iy);
    _mul_cx_real_sub(x + ix, s, y + iy);

    ix += incx * 2;
    iy += incy * 2;
  }
}

void cblas_srot(const int n, float *x, const int incx, float *y, const int incy,
                const float c, const float s) {
  _rot_kernel(n, x, incx, y, incy, c, s);
};

void cblas_drot(const int n, double *x, const int incx, double *y,
                const int incy, const double c, const double s) {
  _rot_kernel(n, x, incx, y, incy, c, s);
}

void cblas_crot(const int n, void *x, const int incx, void *y, const int incy,
                const float c, const void *s) {
  _cs_rot_kernel(n, reinterpret_cast<float *>(x), incx,
                 reinterpret_cast<float *>(y), incy, c,
                 reinterpret_cast<const float *>(s));
}

void cblas_zrot(const int n, void *x, const int incx, void *y, const int incy,
                const double c, const void *s) {
  _cs_rot_kernel(n, reinterpret_cast<double *>(x), incx,
                 reinterpret_cast<double *>(y), incy, c,
                 reinterpret_cast<const double *>(s));
}

void cblas_csrot(const int n, void *x, const int incx, void *y, const int incy,
                 const float c, const float s) {
  _crot_kernel(n, reinterpret_cast<float *>(x), incx,
               reinterpret_cast<float *>(y), incy, c, s);
}

void cblas_zdrot(const int n, void *x, const int incx, void *y, const int incy,
                 const double c, const double s) {
  _crot_kernel(n, reinterpret_cast<double *>(x), incx,
               reinterpret_cast<double *>(y), incy, c, s);
}

}  // namespace bah
