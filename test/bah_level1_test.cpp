#include <cblas.h>
#include <gtest/gtest.h>

#include <complex>

#define BAH_NO_C_API
#include "array_helpers.hpp"
#include "bah.hpp"
#include "print_helpers.hpp"

template <typename RealTp>
RealTp pct_err(RealTp a, RealTp b) {
  return std::fabs(a - b) / ((a + b) / 2);
}

#define EXPECT_FLOAT_PCT_ERR_LT(a, b, err) EXPECT_LT(pct_err(a, b), err)

TEST(Level1_asum, cblas_sasum) {
  const auto x = bah::array::random<float>(512);
  const auto t = cblas_sasum(x.size(), x.data(), 1);
  const auto v = bah::cblas_sasum(x.size(), x.data(), 1);
  std::cout << pct_err(t, v) << "\n";
  EXPECT_FLOAT_PCT_ERR_LT(t, v, 1e-6);
}

TEST(Level1_asum, cblas_dasum) {
  const auto x = bah::array::random<double>(512);
  const auto t = cblas_dasum(x.size(), x.data(), 1);
  const auto v = bah::cblas_dasum(x.size(), x.data(), 1);
  std::cout << pct_err(t, v) << "\n";
  EXPECT_FLOAT_PCT_ERR_LT(t, v, 1e-6);
}

TEST(Level1_asum, cblas_scasum) {
  const auto x = bah::array::random<std::complex<float>>(512);
  const void* xptr = x.data();
  const auto t = cblas_scasum(x.size(), xptr, 1);
  const auto v = bah::cblas_scasum(x.size(), xptr, 1);
  std::cout << pct_err(t, v) << "\n";
  // EXPECT_FLOAT_EQ(t, v);
  EXPECT_FLOAT_PCT_ERR_LT(t, v, 1e-6);
}

TEST(Level1_asum, cblas_dzasum) {
  const auto x = bah::array::random<std::complex<double>>(512);
  const void* xptr = x.data();
  const auto t = cblas_dzasum(x.size(), xptr, 1);
  const auto v = bah::cblas_dzasum(x.size(), xptr, 1);
  std::cout << pct_err(t, v) << "\n";
  // EXPECT_DOUBLE_EQ(t, v);
  EXPECT_FLOAT_PCT_ERR_LT(t, v, 1e-6);
}
