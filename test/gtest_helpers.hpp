#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

namespace helpers {

template <typename RealTp>
RealTp pct_err(const RealTp a, const RealTp b) {
  return std::fabs(a - b) / ((a + b) / 2);
}

template <typename RealTp>
void EXPECT_FLOAT_PCT_ERR_LT(const RealTp a, const RealTp b, const RealTp err) {
  EXPECT_LT(pct_err(a, b), err);
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
