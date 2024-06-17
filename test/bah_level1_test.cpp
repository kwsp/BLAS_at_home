#include <cblas.h>
#include <gtest/gtest.h>

#include <array>
#include <complex>

#define BAH_NO_C_API
#include "bah.hpp"
#include "bah_array.hpp"
#include "bah_concepts.hpp"
#include "gtest_helpers.hpp"
#include "print_helpers.hpp"

constexpr std::array STRIDES{1, 2, 3, 4, 8};

using bah::cdouble;
using bah::cfloat;
using namespace bah::concepts;

template <is_real DTYPE, is_function F>
void test_asum(const F& f_t, const F& f_v, DTYPE tol) {
  const auto x = bah::Arr<DTYPE>::random(512);
  const auto _f = [&](const int incx) {
    const auto t = f_t(x.size(), x.data(), incx);
    const auto v = f_v(x.size(), x.data(), incx);
    helpers::EXPECT_FLOAT_PCT_ERR_LT(t, v, tol);
  };
  for (const auto stride : STRIDES) _f(stride);
}

template <is_complex DTYPE, is_function F>
void test_asum(const F& f_t, const F& f_v, typename DTYPE::value_type tol) {
  const auto x = bah::Arr<DTYPE>::random(512);
  const auto _f = [&](const int incx) {
    const auto t = f_t(x.size(), x.data(), incx);
    const auto v = f_v(x.size(), x.data(), incx);
    helpers::EXPECT_FLOAT_PCT_ERR_LT(t, v, tol);
  };
  for (const auto stride : STRIDES) _f(stride);
}

TEST(Level1_asum, cblas_sasum) {
  test_asum(cblas_sasum, bah::cblas_sasum, 1e-6f);
}
TEST(Level1_asum, cblas_dasum) {
  test_asum(cblas_dasum, bah::cblas_dasum, 1e-9);
}
TEST(Level1_asum, cblas_scasum) {
  test_asum<cfloat>(cblas_scasum, bah::cblas_scasum, 1e-6f);
}
TEST(Level1_asum, cblas_dzasum) {
  test_asum<cdouble>(cblas_dzasum, bah::cblas_dzasum, 1e-6f);
}

template <is_real DTYPE, is_function F>
void test_axpy(F& f_t, F& f_v) {
  const auto x = bah::Arr<DTYPE>::random(512);
  const DTYPE a{1.234};
  const auto _f = [&](const int incx, const int incy) {
    std::vector<DTYPE> y_t(x.size(), 0);
    const int n = std::min(x.size() / incx, y_t.size() / incy);

    f_t(n, a, x.data(), incx, y_t.data(), incy);

    std::vector<DTYPE> y_v(x.size(), 0);
    f_v(n, a, x.data(), incx, y_v.data(), incy);

    helpers::ASSERT_VEC_EQ(y_t, y_v);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

TEST(Level1_axpy, cblas_saxpy) {
  test_axpy<float>(cblas_saxpy, bah::cblas_saxpy);
}
TEST(Level1_axpy, cblas_daxpy) {
  test_axpy<double>(cblas_daxpy, bah::cblas_daxpy);
}

template <is_complex DTYPE, is_function F>
void test_axpy_cx(F& f_t, F& f_v) {
  const auto x = bah::Arr<DTYPE>::random(512);
  const DTYPE a{1.234, 2.345};
  const auto _f = [&](const int incx, const int incy) {
    std::vector<DTYPE> y_t(x.size(), 0);
    const int n = std::min(x.size() / incx, y_t.size() / incy);

    f_t(n, &a, x.data(), incx, y_t.data(), incy);
    std::vector<DTYPE> y_v(x.size(), 0);
    f_v(n, &a, x.data(), incx, y_v.data(), incy);

    helpers::ASSERT_VEC_EQ(y_t, y_v);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

TEST(Level1_axpy, cblas_caxpy) {
  test_axpy_cx<cfloat>(cblas_caxpy, bah::cblas_caxpy);
}
TEST(Level1_axpy, cblas_zaxpy) {
  test_axpy_cx<cdouble>(cblas_zaxpy, bah::cblas_zaxpy);
}

template <typename DTYPE, is_function F>
void test_copy(F& f_t, F& f_v) {
  const auto x = bah::Arr<DTYPE>::arange(512);
  const auto _f = [&](const int incx, const int incy) {
    std::vector<DTYPE> y_t(x.size(), 0);
    const int n = std::min(x.size() / incx, y_t.size() / incy);

    f_t(n, x.data(), incx, y_t.data(), incy);

    std::vector<DTYPE> y_v(x.size(), 0);
    f_v(n, x.data(), incx, y_v.data(), incy);

    helpers::ASSERT_VEC_EQ(y_t, y_v);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

TEST(Level1_copy, cblas_scopy) {
  test_copy<float>(cblas_scopy, bah::cblas_scopy);
}
TEST(Level1_copy, cblas_dcopy) {
  test_copy<double>(cblas_dcopy, bah::cblas_dcopy);
}
TEST(Level1_copy, cblas_ccopy) {
  test_copy<cfloat>(cblas_ccopy, bah::cblas_ccopy);
}
TEST(Level1_copy, cblas_zcopy) {
  test_copy<cdouble>(cblas_zcopy, bah::cblas_zcopy);
}

template <is_real DTYPE, is_function F>
void test_dot(F& f_t, F& f_v, DTYPE tol) {
  const auto x = bah::Arr<DTYPE>::random(512);
  const auto y = bah::Arr<DTYPE>::random(512);

  const auto _f = [&](const int incx, const int incy) {
    const int n = std::min(x.size() / incx, y.size() / incy);
    const auto t = f_t(n, x.data(), incx, y.data(), incy);
    const auto v = f_v(n, x.data(), incx, y.data(), incy);
    helpers::EXPECT_FLOAT_PCT_ERR_LT(t, v, tol);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

TEST(Level1_dot, cblas_sdot) {
  test_dot<float>(cblas_sdot, bah::cblas_sdot, 1e-5f);
}
TEST(Level1_dot, cblas_ddot) {
  test_dot<double>(cblas_ddot, bah::cblas_ddot, 1e-9);
}

TEST(Level1_sdot, cblas_sdsdot) {
  const auto x = bah::Arr<float>::arange(64);
  const auto y = bah::Arr<float>::arange(64);

  const auto _f = [&](const int incx, const int incy) {
    const int n = std::min(x.size() / incx, y.size() / incy);
    const float sb = 1.14;
    const auto t = cblas_sdsdot(n, sb, x.data(), incx, y.data(), incy);
    const auto v = bah::cblas_sdsdot(n, sb, x.data(), incx, y.data(), incy);
    helpers::EXPECT_FLOAT_PCT_ERR_LT(t, v, 1e-5f);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

TEST(Level1_sdot, cblas_dsdot) {
  const auto x = bah::Arr<float>::arange(64);
  const auto y = bah::Arr<float>::arange(64);

  const auto _f = [&](const int incx, const int incy) {
    const int n = std::min(x.size() / incx, y.size() / incy);
    const auto t = cblas_dsdot(n, x.data(), incx, y.data(), incy);
    const auto v = bah::cblas_dsdot(n, x.data(), incx, y.data(), incy);
    helpers::EXPECT_FLOAT_PCT_ERR_LT(t, v, 1e-6);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

template <is_complex DTYPE, is_function F>
void test_complex_dot(const F& f_t, const F& f_v,
                      typename DTYPE::value_type tol) {
  const auto x = bah::Arr<DTYPE>::random(512);
  const auto y = bah::Arr<DTYPE>::random(512);
  const auto _f = [&](const int incx, const int incy) {
    const int n = std::min(x.size() / incx, y.size() / incy);
    DTYPE t{};
    f_t(n, x.data(), incx, y.data(), incy, &t);
    DTYPE v{};
    f_v(n, x.data(), incx, y.data(), incy, &v);
    helpers::EXPECT_FLOAT_PCT_ERR_LT(t, v, tol);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

TEST(Level1_dotc, cblas_cdotc_sub) {
  test_complex_dot<cfloat>(cblas_cdotc_sub, bah::cblas_cdotc_sub, 1e-3f);
}
TEST(Level1_dotc, cblas_zdotc_sub) {
  test_complex_dot<cdouble>(cblas_zdotc_sub, bah::cblas_zdotc_sub, 1e-9);
}

template <is_real DTYPE, is_function F>
void test_nrm2(const F& f_t, const F& f_v, DTYPE tol) {
  const auto x = bah::Arr<DTYPE>::random(8);
  const auto _f = [&](const int incx) {
    const int n = x.size() / incx;
    const auto t = f_t(n, x.data(), incx);
    const auto v = f_v(n, x.data(), incx);
    helpers::EXPECT_FLOAT_PCT_ERR_LT(t, v, tol);
  };
  for (const auto stride_x : STRIDES) _f(stride_x);
}

template <is_complex DTYPE, is_function F>
void test_nrm2_cx(const F& f_t, const F& f_v, typename DTYPE::value_type tol) {
  const auto x = bah::Arr<DTYPE>::random(8);
  const auto _f = [&](const int incx) {
    const int n = x.size() / incx;
    const auto t = f_t(n, x.data(), incx);
    const auto v = f_v(n, x.data(), incx);
    helpers::EXPECT_FLOAT_PCT_ERR_LT(t, v, tol);
  };
  for (const auto stride_x : STRIDES) _f(stride_x);
}

TEST(Level1_nrm2, cblas_snrm2) {
  test_nrm2(cblas_snrm2, bah::cblas_snrm2, 1e-6f);
}
TEST(Level1_nrm2, cblas_dnrm2) {
  test_nrm2(cblas_dnrm2, bah::cblas_dnrm2, 1e-9);
}
TEST(Level1_nrm2, cblas_scnrm2) {
  test_nrm2(cblas_scnrm2, bah::cblas_scnrm2, 1e-6f);
}
TEST(Level1_nrm2, cblas_dznrm2) {
  test_nrm2(cblas_dznrm2, bah::cblas_dznrm2, 1e-9);
}

// Test rot for real x, y, c, s
template <typename DTYPE, is_function F>
void test_rot(const F& f_t, const F& f_v, DTYPE tol, DTYPE c, DTYPE s) {
  const auto _f = [&](const int incx, const int incy) {
    auto x_t = bah::Arr<DTYPE>::arange(8);
    auto y_t = bah::Arr<DTYPE>::arange(8);
    const int n = std::min(x_t.size() / incx, y_t.size() / incy);

    auto x_v = bah::Arr<DTYPE>::arange(8);
    auto y_v = bah::Arr<DTYPE>::arange(8);

    f_t(n, x_t.data(), incx, y_t.data(), incy, c, s);
    f_v(n, x_v.data(), incx, y_v.data(), incy, c, s);

    helpers::ASSERT_FLOAT_VEC_EQ(x_t, x_v, tol);
    helpers::ASSERT_FLOAT_VEC_EQ(y_t, y_v, tol);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

// Test rot for complex x, y, s, real c
template <is_complex DTYPE, is_function F>
void test_rot_cx_real(const F& f_t, const F& f_v,
                      typename DTYPE::value_type tol,
                      typename DTYPE::value_type c,
                      typename DTYPE::value_type* s) {
  const auto _f = [&](const int incx, const int incy) {
    auto x_t = bah::Arr<DTYPE>::arange(8);
    auto y_t = bah::Arr<DTYPE>::arange(8);
    const int n = std::min(x_t.size() / incx, y_t.size() / incy);

    auto x_v = bah::Arr<DTYPE>::arange(8);
    auto y_v = bah::Arr<DTYPE>::arange(8);

    f_t(n, x_t.data(), incx, y_t.data(), incy, c, s);
    f_v(n, x_v.data(), incx, y_v.data(), incy, c, s);

    helpers::ASSERT_FLOAT_VEC_EQ(x_t, x_v, tol);
    helpers::ASSERT_FLOAT_VEC_EQ(y_t, y_v, tol);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

template <is_complex DTYPE, is_function F>
void test_rot_cx(const F& f_t, const F& f_v, typename DTYPE::value_type tol,
                 typename DTYPE::value_type c, typename DTYPE::value_type s) {
  const auto _f = [&](const int incx, const int incy) {
    auto x_t = bah::Arr<DTYPE>::arange(8);
    auto y_t = bah::Arr<DTYPE>::arange(8);
    const int n = std::min(x_t.size() / incx, y_t.size() / incy);

    auto x_v = bah::Arr<DTYPE>::arange(8);
    auto y_v = bah::Arr<DTYPE>::arange(8);

    f_t(n, x_t.data(), incx, y_t.data(), incy, c, s);
    f_v(n, x_v.data(), incx, y_v.data(), incy, c, s);

    helpers::ASSERT_FLOAT_VEC_EQ(x_t, x_v, tol);
    helpers::ASSERT_FLOAT_VEC_EQ(y_t, y_v, tol);
  };
  for (const auto stride_x : STRIDES) {
    for (const auto stride_y : STRIDES) {
      _f(stride_x, stride_y);
    }
  }
}

TEST(Level1_rot, cblas_srot) {
  test_rot<float>(cblas_srot, bah::cblas_srot, 1e-6f, 1.89f, 3.2f);
}
TEST(Level1_rot, cblas_drot) {
  test_rot<double>(cblas_drot, bah::cblas_drot, 1e-9, 2.33, 0.53);
}

TEST(Level1_rot, cblas_crot) {
  float s[2] = {0.53, 0.32};
  test_rot_cx_real<cfloat>(cblas_crot, bah::cblas_crot, 1e-9, 2.33, s);
}
// TEST(Level1_rot, cblas_zrot) {
// test_rot<double>(cblas_drot, bah::cblas_drot, 1e-9, 2.33, 0.53);
//}

//_TEST_ROT(cblas_crot, cfloat, 1e-6f, float, 2.33, cfloat, cfloat(0.53,
// 0.12));

TEST(Level1_rot, cblas_csrot) {
  test_rot_cx<cfloat>(cblas_csrot, bah::cblas_csrot, 1e-6f, 2.33, 0.53);
}
TEST(Level1_rot, cblas_zdrot) {
  test_rot_cx<cdouble>(cblas_zdrot, bah::cblas_zdrot, 1e-9, 2.33, 0.53);
}
