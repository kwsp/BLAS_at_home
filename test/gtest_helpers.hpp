#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

namespace helpers {

template <typename T>
void ASSERT_VEC_EQ(const std::vector<T>& a, const std::vector<T>& b) {
  ASSERT_EQ(a.size(), b.size());
  for (int i = 0; i < a.size(); i++) {
    EXPECT_EQ(a[i], b[i]);
  }
}

}  // namespace helpers
