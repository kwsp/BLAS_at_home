#include "bah_array.hpp"

#include <gtest/gtest.h>

using namespace bah::arr;

TEST(bah_array, arange) {
  Arr b = Arr<float>::arange(10);
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(b[i], i);
  }
}

TEST(bah_array, random) {
  // Test compiles fine
  Arr a = Arr<float>::random(512);
}
