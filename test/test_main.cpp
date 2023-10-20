#include <cblas.h>

#include <cstdio>
#include <iostream>
#include <vector>

#include "array_helpers.hpp"
#include "bah.hpp"
#include "print_helpers.hpp"

void run_level1() {
  auto x = bah::array::random<float>(100, -1.0f, 1.0f);
  std::cout << x;
  std::cout << "     sasum " << cblas_sasum(x.size(), x.data(), 1) << "\n";
  std::cout << "bah::sasum " << bah::cblas_sasum(x.size(), x.data(), 1) << "\n";
}

int main() {
  run_level1();
  return 0;
}
