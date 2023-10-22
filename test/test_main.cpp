#include <cblas.h>

#include <cassert>
#include <complex>
#include <cstdio>
#include <iostream>
#include <vector>

#include "array_helpers.hpp"
#define BAH_NO_C_API
#include "bah.hpp"
#include "print_helpers.hpp"
#include "timer.hpp"

void run_level1() {
  {
    // auto x = bah::array::random<float>(1024, -1.0f, 1.0f);
    auto x = bah::array::arange<float>(1024);
    std::cout << x;
    std::cout << "     sasum " << cblas_sasum(x.size(), x.data(), 1) << "\n";
    std::cout << "bah::sasum " << bah::cblas_sasum(x.size(), x.data(), 1)
              << "\n";

    bah::timer::timeit([&]() { bah::cblas_sasum(x.size(), x.data(), 1); },
                       100000);
    bah::timer::timeit([&]() { cblas_sasum(x.size(), x.data(), 1); }, 100000);
  }

  {
    auto x = bah::array::random<std::complex<float>>(5);
    std::cout << x;
  }
}

int main() {
  run_level1();
  return 0;
}
