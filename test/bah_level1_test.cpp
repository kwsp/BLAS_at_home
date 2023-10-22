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

TEST(Level1_asum, cblas_sasum) {
  const auto x = bah::array::random<float>(512);
  const auto _f = [&](const int incx) {
    const auto t = cblas_sasum(x.size(), x.data(), incx);
    const auto v = bah::cblas_sasum(x.size(), x.data(), incx);
    helpers::EXPECT_FLOAT_PCT_ERR_LT(t, v, 1e-6f);
  };
  for (const auto stride : STRIDES) _f(stride);
}

TEST(Level1_asum, cblas_dasum) {
  const auto x = bah::array::random<double>(512);
  const auto _f = [&](const int incx) {
    const auto t = cblas_dasum(x.size(), x.data(), incx);
    const auto v = bah::cblas_dasum(x.size(), x.data(), incx);
    helpers::EXPECT_FLOAT_PCT_ERR_LT(t, v, 1e-6);
  };
  for (const auto stride : STRIDES) _f(stride);
}

TEST(Level1_asum, cblas_scasum) {
  const auto x = bah::array::random<cfloat>(512);
  const void* xptr = x.data();
  const auto _f = [&](const int incx) {
    const auto t = cblas_scasum(x.size(), xptr, incx);
    const auto v = bah::cblas_scasum(x.size(), xptr, incx);
    helpers::EXPECT_FLOAT_PCT_ERR_LT(t, v, 1e-6f);
  };
  for (const auto stride : STRIDES) _f(stride);
}

TEST(Level1_asum, cblas_dzasum) {
  const auto x = bah::array::random<cdouble>(512);
  const void* xptr = x.data();
  const auto _f = [&](const int incx) {
    const auto t = cblas_dzasum(x.size(), xptr, incx);
    const auto v = bah::cblas_dzasum(x.size(), xptr, incx);
    helpers::EXPECT_FLOAT_PCT_ERR_LT(t, v, 1e-6);
  };
  for (const auto stride : STRIDES) _f(stride);
}

TEST(Level1_axpy, cblas_saxpy) {
  const std::vector<float> x = bah::array::random<float>(512);
  const float a{1.234};

  const auto _f = [&](const int incx, const int incy) {
    std::vector<float> y_t(x.size(), 0);
    const int n = std::min(x.size() / incx, y_t.size() / incy);

    cblas_saxpy(n, a, x.data(), incx, y_t.data(), incy);

    std::vector<float> y_v(x.size(), 0);
    bah::cblas_saxpy(n, a, x.data(), incx, y_v.data(), incy);

    helpers::ASSERT_VEC_EQ(y_t, y_v);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

TEST(Level1_axpy, cblas_daxpy) {
  {
    const std::vector<double> x = bah::array::random<double>(512);
    const double a{1.234};

    const auto _f = [&](const int incx, const int incy) {
      std::vector<double> y_t(x.size(), 0);
      const int n = std::min(x.size() / incx, y_t.size() / incy);

      cblas_daxpy(n, a, x.data(), incx, y_t.data(), incy);

      std::vector<double> y_v(x.size(), 0);
      bah::cblas_daxpy(n, a, x.data(), incx, y_v.data(), incy);

      helpers::ASSERT_VEC_EQ(y_t, y_v);
    };
    for (const auto stride_x : STRIDES) {
      for (const auto stride_y : STRIDES) {
        _f(stride_x, stride_y);
      }
    }
  }
}

TEST(Level1_axpy, cblas_caxpy) {
  using T = cfloat;
  const std::vector<T> x = bah::array::random<T>(512);
  const T a{1.234, 2.345};

  const auto _f = [&](const int incx, const int incy) {
    std::vector<T> y_t(x.size(), 0);
    const int n = std::min(x.size() / incx, y_t.size() / incy);
    cblas_caxpy(n, &a, x.data(), incx, y_t.data(), incy);

    std::vector<T> y_v(x.size(), 0);
    bah::cblas_caxpy(n, &a, x.data(), incx, y_v.data(), incy);

    helpers::ASSERT_VEC_EQ(y_t, y_v);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

TEST(Level1_axpy, cblas_zaxpy) {
  using T = cdouble;
  const std::vector<T> x = bah::array::random<T>(512);
  const T a{1.234, 2.345};

  const auto _f = [&](const int incx, const int incy) {
    std::vector<T> y_t(x.size(), 0);
    const int n = std::min(x.size() / incx, y_t.size() / incy);
    cblas_zaxpy(n, &a, x.data(), incx, y_t.data(), incy);

    std::vector<T> y_v(x.size(), 0);
    bah::cblas_zaxpy(n, &a, x.data(), incx, y_v.data(), incy);

    helpers::ASSERT_VEC_EQ(y_t, y_v);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

TEST(Level1_copy, cblas_scopy) {
  using T = float;
  const auto x = bah::array::arange<T>(512);

  const auto _f = [&](const int incx, const int incy) {
    std::vector<T> y_t(x.size(), 0);
    const int n = std::min(x.size() / incx, y_t.size() / incy);
    cblas_scopy(n, x.data(), incx, y_t.data(), incy);

    std::vector<T> y_v(x.size(), 0);
    bah::cblas_scopy(n, x.data(), incx, y_v.data(), incy);

    helpers::ASSERT_VEC_EQ(y_t, y_v);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}
TEST(Level1_copy, cblas_dcopy) {
  using T = double;
  const auto x = bah::array::arange<T>(512);

  const auto _f = [&](const int incx, const int incy) {
    std::vector<T> y_t(x.size(), 0);
    const int n = std::min(x.size() / incx, y_t.size() / incy);
    cblas_dcopy(n, x.data(), incx, y_t.data(), incy);

    std::vector<T> y_v(x.size(), 0);
    bah::cblas_dcopy(n, x.data(), incx, y_v.data(), incy);

    helpers::ASSERT_VEC_EQ(y_t, y_v);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

TEST(Level1_copy, cblas_ccopy) {
  using T = cfloat;
  const auto x = bah::array::arange<T>(512);

  const auto _f = [&](const int incx, const int incy) {
    std::vector<T> y_t(x.size(), 0);
    const int n = std::min(x.size() / incx, y_t.size() / incy);
    cblas_ccopy(n, x.data(), incx, y_t.data(), incy);

    std::vector<T> y_v(x.size(), 0);
    bah::cblas_ccopy(n, x.data(), incx, y_v.data(), incy);

    helpers::ASSERT_VEC_EQ(y_t, y_v);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

TEST(Level1_copy, cblas_zcopy) {
  using T = cdouble;
  const auto x = bah::array::arange<T>(512);

  const auto _f = [&](const int incx, const int incy) {
    std::vector<T> y_t(x.size(), 0);
    const int n = std::min(x.size() / incx, y_t.size() / incy);
    cblas_zcopy(n, x.data(), incx, y_t.data(), incy);

    std::vector<T> y_v(x.size(), 0);
    bah::cblas_zcopy(n, x.data(), incx, y_v.data(), incy);

    helpers::ASSERT_VEC_EQ(y_t, y_v);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

TEST(Level1_dot, cblas_sdot) {
  const auto x = bah::array::arange<float>(512);
  const auto y = bah::array::arange<float>(512);

  const auto _f = [&](const int incx, const int incy) {
    const int n = std::min(x.size() / incx, y.size() / incy);
    const auto t = cblas_sdot(x.size(), x.data(), incx, y.data(), incy);
    const auto v = bah::cblas_sdot(x.size(), x.data(), incx, y.data(), incy);
    helpers::EXPECT_FLOAT_PCT_ERR_LT(t, v, 1e-5f);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

TEST(Level1_dot, cblas_ddot) {
  const auto x = bah::array::arange<double>(512);
  const auto y = bah::array::arange<double>(512);

  const auto _f = [&](const int incx, const int incy) {
    const int n = std::min(x.size() / incx, y.size() / incy);
    const auto t = cblas_ddot(x.size(), x.data(), incx, y.data(), incy);
    const auto v = bah::cblas_ddot(x.size(), x.data(), incx, y.data(), incy);
    helpers::EXPECT_FLOAT_PCT_ERR_LT(t, v, 1e-6);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}
