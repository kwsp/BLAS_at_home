#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <complex>
#include <numeric>
#include <vector>

namespace helpers {

template <typename RealTp>
RealTp pct_err(const RealTp a, const RealTp b) {
  if (a == 0 && b == 0) return 0.;
  if (a == 0 || b == 0) return 1.;
  return std::fabs(a - b) / ((std::fabs(a) + std::fabs(b)) / 2);
}

template <typename RealTp>
void EXPECT_FLOAT_PCT_ERR_LT(const RealTp a, const RealTp b, const RealTp err) {
  EXPECT_LT(pct_err(a, b), err) << "a = " << a << ", b = " << b << "\n";
}

template <typename RealTp>
void EXPECT_CX_PCT_ERR_LT(const std::complex<RealTp> a,
                          const std::complex<RealTp> b, const RealTp err) {
  auto real_err = pct_err(a.real(), b.real());
  auto imag_err = pct_err(a.imag(), b.imag());
  EXPECT_LT(real_err + imag_err, err * 2)
      << "a = " << a << ", b = " << b << "\n";
}

template <typename T>
void ASSERT_VEC_EQ(const std::vector<T>& a, const std::vector<T>& b) {
  ASSERT_EQ(a.size(), b.size());
  for (int i = 0; i < a.size(); i++) {
    EXPECT_EQ(a[i], b[i]);
  }
}

template <typename T>
void ASSERT_FLOAT_VEC_EQ(const std::vector<T>& a, const std::vector<T>& b,
                         const T err) {
  ASSERT_EQ(a.size(), b.size());
  for (int i = 0; i < a.size(); i++) {
    EXPECT_FLOAT_PCT_ERR_LT(a[i], b[i], err);
  }
}

}  // namespace helpers
