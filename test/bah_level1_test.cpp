#include <cblas.h>
#include <gtest/gtest.h>

#include <array>
#include <complex>

#define BAH_NO_C_API
#include "array_helpers.hpp"
#include "bah.hpp"
#include "gtest_helpers.hpp"
#include "print_helpers.hpp"

using cfloat = std::complex<float>;
using cdouble = std::complex<double>;

const std::array STRIDES{1, 2, 3, 4, 8, 16};

#define _TEST_ASUM(FUNC, DTYPE, TOL)                      \
  TEST(Level1_asum, FUNC) {                               \
    const auto x = bah::array::random<DTYPE>(512);        \
    const auto _f = [&](const int incx) {                 \
      const auto t = FUNC(x.size(), x.data(), incx);      \
      const auto v = bah::FUNC(x.size(), x.data(), incx); \
      helpers::EXPECT_FLOAT_PCT_ERR_LT(t, v, TOL);        \
    };                                                    \
    for (const auto stride : STRIDES) _f(stride);         \
  }

_TEST_ASUM(cblas_sasum, float, 1e-6f);
_TEST_ASUM(cblas_dasum, double, 1e-9);
_TEST_ASUM(cblas_scasum, cfloat, 1e-6f);
_TEST_ASUM(cblas_dzasum, cdouble, 1e-9);

#define _TEST_AXPY(FUNC, DTYPE)                                   \
  TEST(Level1_axpy, FUNC) {                                       \
    const auto x = bah::array::random<DTYPE>(512);                \
    const DTYPE a{1.234};                                         \
    const auto _f = [&](const int incx, const int incy) {         \
      std::vector<DTYPE> y_t(x.size(), 0);                        \
      const int n = std::min(x.size() / incx, y_t.size() / incy); \
                                                                  \
      FUNC(n, a, x.data(), incx, y_t.data(), incy);               \
      std::vector<DTYPE> y_v(x.size(), 0);                        \
      bah::FUNC(n, a, x.data(), incx, y_v.data(), incy);          \
                                                                  \
      helpers::ASSERT_VEC_EQ(y_t, y_v);                           \
    };                                                            \
    for (const auto stride_x : STRIDES) {                         \
      for (const auto stride_y : STRIDES) {                       \
        _f(stride_x, stride_y);                                   \
      }                                                           \
    }                                                             \
  }

_TEST_AXPY(cblas_saxpy, float);
_TEST_AXPY(cblas_daxpy, double);

#define _TEST_AXPY_CX(FUNC, DTYPE)                                \
  TEST(Level1_axpy, FUNC) {                                       \
    const auto x = bah::array::random<DTYPE>(512);                \
    const DTYPE a{1.234, 2.345};                                  \
    const auto _f = [&](const int incx, const int incy) {         \
      std::vector<DTYPE> y_t(x.size(), 0);                        \
      const int n = std::min(x.size() / incx, y_t.size() / incy); \
                                                                  \
      FUNC(n, &a, x.data(), incx, y_t.data(), incy);              \
      std::vector<DTYPE> y_v(x.size(), 0);                        \
      bah::FUNC(n, &a, x.data(), incx, y_v.data(), incy);         \
                                                                  \
      helpers::ASSERT_VEC_EQ(y_t, y_v);                           \
    };                                                            \
    for (const auto stride_x : STRIDES) {                         \
      for (const auto stride_y : STRIDES) {                       \
        _f(stride_x, stride_y);                                   \
      }                                                           \
    }                                                             \
  }

_TEST_AXPY_CX(cblas_caxpy, cfloat);
_TEST_AXPY_CX(cblas_zaxpy, cdouble);

#define _TEST_COPY(FUNC, DTYPE)                                   \
  TEST(Level1_copy, FUNC) {                                       \
    const auto x = bah::array::arange<DTYPE>(512);                \
    const auto _f = [&](const int incx, const int incy) {         \
      std::vector<DTYPE> y_t(x.size(), 0);                        \
      const int n = std::min(x.size() / incx, y_t.size() / incy); \
      FUNC(n, x.data(), incx, y_t.data(), incy);                  \
                                                                  \
      std::vector<DTYPE> y_v(x.size(), 0);                        \
      bah::FUNC(n, x.data(), incx, y_v.data(), incy);             \
                                                                  \
      helpers::ASSERT_VEC_EQ(y_t, y_v);                           \
    };                                                            \
    for (const auto stride_x : STRIDES) {                         \
      for (const auto stride_y : STRIDES) {                       \
        _f(stride_x, stride_y);                                   \
      }                                                           \
    }                                                             \
  }

_TEST_COPY(cblas_scopy, float);
_TEST_COPY(cblas_dcopy, double);
_TEST_COPY(cblas_ccopy, cfloat);
_TEST_COPY(cblas_zcopy, cdouble);

#define _TEST_DOT(FUNC, DTYPE, TOL)                                       \
  TEST(Level1_dot, FUNC) {                                                \
    const auto x = bah::array::random<DTYPE>(512);                        \
    const auto y = bah::array::random<DTYPE>(512);                        \
                                                                          \
    const auto _f = [&](const int incx, const int incy) {                 \
      const int n = std::min(x.size() / incx, y.size() / incy);           \
      const auto t = FUNC(x.size(), x.data(), incx, y.data(), incy);      \
      const auto v = bah::FUNC(x.size(), x.data(), incx, y.data(), incy); \
      helpers::EXPECT_FLOAT_PCT_ERR_LT(t, v, TOL);                        \
    };                                                                    \
    for (const auto stride_x : STRIDES) {                                 \
      for (const auto stride_y : STRIDES) {                               \
        _f(stride_x, stride_y);                                           \
      }                                                                   \
    }                                                                     \
  }

_TEST_DOT(cblas_sdot, float, 1e-5f);
_TEST_DOT(cblas_ddot, double, 1e-9);

TEST(Level1_sdot, cblas_sdsdot) {
  const auto x = bah::array::arange<float>(64);
  const auto y = bah::array::arange<float>(64);

  const auto _f = [&](const int incx, const int incy) {
    const int n = std::min(x.size() / incx, y.size() / incy);
    const float sb = 1.14;
    const auto t = cblas_sdsdot(x.size(), sb, x.data(), incx, y.data(), incy);
    const auto v =
        bah::cblas_sdsdot(x.size(), sb, x.data(), incx, y.data(), incy);
    helpers::EXPECT_FLOAT_PCT_ERR_LT(t, v, 1e-5f);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

TEST(Level1_sdot, cblas_dsdot) {
  const auto x = bah::array::arange<float>(64);
  const auto y = bah::array::arange<float>(64);

  const auto _f = [&](const int incx, const int incy) {
    const int n = std::min(x.size() / incx, y.size() / incy);
    const auto t = cblas_dsdot(x.size(), x.data(), incx, y.data(), incy);
    const auto v = bah::cblas_dsdot(x.size(), x.data(), incx, y.data(), incy);
    helpers::EXPECT_FLOAT_PCT_ERR_LT(t, v, 1e-6);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

#define _TEST_COMPLEX_DOT(FUNC, DType, tol)                     \
  TEST(Level1_dotc, FUNC) {                                     \
    const auto x = bah::array::random<DType>(512);              \
    const auto y = bah::array::random<DType>(512);              \
    const auto _f = [&](const int incx, const int incy) {       \
      const int n = std::min(x.size() / incx, y.size() / incy); \
      DType t{};                                                \
      FUNC(x.size(), x.data(), incx, y.data(), incy, &t);       \
      DType v{};                                                \
      bah::FUNC(x.size(), x.data(), incx, y.data(), incy, &v);  \
      helpers::EXPECT_CX_PCT_ERR_LT(t, v, tol);                 \
    };                                                          \
    for (const auto stride_x : STRIDES) {                       \
      for (const auto stride_y : STRIDES) {                     \
        _f(stride_x, stride_y);                                 \
      }                                                         \
    }                                                           \
  }

_TEST_COMPLEX_DOT(cblas_cdotc_sub, cfloat, 1e-3f)
_TEST_COMPLEX_DOT(cblas_zdotc_sub, cdouble, 1e-9)
