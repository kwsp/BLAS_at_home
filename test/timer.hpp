#pragma once

#include <chrono>
#include <functional>

#include "print_helpers.hpp"

namespace bah::timer {

inline auto timeit(const std::function<void()>& f, int n = 10) {
  using namespace std::chrono;
  const auto start = high_resolution_clock::now();

  for (int i = 0; i < n; i++) f();

  const auto elapsed = high_resolution_clock::now() - start;
  const auto ms =
      static_cast<double>(duration_cast<milliseconds>(elapsed).count()) / n;

  std::cout << "Took " << ms << "ms\n";

  return ms;
}

}  // namespace bah::timer
