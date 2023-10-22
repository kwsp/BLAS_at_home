#pragma once

#include <cmath>
#include <complex>
#include <format>
#include <iostream>
#include <ranges>

#include "array_helpers.hpp"

// Print vector like
// (#5) [1, 2, 3, 4, 5]
template <class T, size_t MAX_N = 25>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
  os << "(#" << vec.size() << ") [";

  if (vec.empty()) {
    os << "]\n";
    return os;
  }

  os << vec.front();
  size_t n = std::min(MAX_N, vec.size());
  for (size_t i = 1; i < n; i++) os << ", " << vec[i];

  if (n != vec.size()) os << ", ...";

  os << "]\n";
  return os;
}

// template <typename T>
// std::ostream &operator<<(std::ostream &os, const std::complex<T>& v) {
// os <<
//}
