#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <complex>
#include <numeric>
#include <vector>

#include "bah_array.hpp"
#include "bah_concepts.hpp"

namespace helpers {
using namespace bah::concepts;

template <typename Real>
Real pct_err(const Real a, const Real b) {
  if (a == 0 && b == 0) return 0.;
  if (a == 0 || b == 0) return 1.;
  return std::fabs(a - b) / ((std::fabs(a) + std::fabs(b)) / 2);
}

template <is_complex Cx>
void EXPECT_FLOAT_PCT_ERR_LT(const Cx a, const Cx b,
                             const typename Cx::value_type tol) {
  auto real_err = pct_err(a.real(), b.real());
  auto imag_err = pct_err(a.imag(), b.imag());
  EXPECT_LT(real_err + imag_err, tol * 2)
      << "a = " << a << ", b = " << b << "\n";
}

template <is_real Real>
void EXPECT_FLOAT_PCT_ERR_LT(const Real a, const Real b, const Real tol) {
  EXPECT_LT(pct_err(a, b), tol) << "a = " << a << ", b = " << b << "\n";
}

template <typename T>
void ASSERT_VEC_EQ(const std::vector<T>& a, const std::vector<T>& b) {
  EXPECT_FLOAT_PCT_ERR_LT(1., 1.0, 1.0);
  ASSERT_EQ(a.size(), b.size());
  for (int i = 0; i < a.size(); i++) {
    EXPECT_EQ(a[i], b[i]);
  }
}

template <is_real T>
void ASSERT_FLOAT_VEC_EQ(const std::vector<T>& a, const std::vector<T>& b,
                         const T tol) {
  ASSERT_EQ(a.size(), b.size());
  for (int i = 0; i < a.size(); i++) {
    EXPECT_FLOAT_PCT_ERR_LT(a[i], b[i], tol);
  }
}

template <is_complex T>
void ASSERT_FLOAT_VEC_EQ(const std::vector<T>& a, const std::vector<T>& b,
                         const typename T::value_type tol) {
  ASSERT_EQ(a.size(), b.size());
  for (int i = 0; i < a.size(); i++) {
    EXPECT_FLOAT_PCT_ERR_LT(a[i], b[i], tol);
  }
}

}  // namespace helpers
