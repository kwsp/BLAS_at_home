#pragma once
// A small template array library

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include "bah_concepts.hpp"

namespace bah::arr {

template <concepts::is_complex Cx>
void _random(std::vector<Cx> &arr, Cx low, Cx high) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis_real(low.real(), high.real());
  std::uniform_real_distribution<> dis_imag(low.imag(), high.imag());

  std::generate(arr.begin(), arr.end(),
                [&]() { return Cx(dis_real(gen), dis_imag(gen)); });
}

template <concepts::is_real Real>
void _random(std::vector<Real> &arr, Real low, Real high) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(low, high);

  std::generate(arr.begin(), arr.end(), [&]() { return dis(gen); });
}

template <typename Real>
void _arange(std::vector<Real> &arr, size_t n) {
  assert(arr.size() >= n);
  std::iota(arr.begin(), arr.begin() + n, 0);
}

template <typename T>
class Arr : public std::vector<T> {
 public:
  // Inherit all constructors
  using std::vector<T>::vector;

  static Arr<T> random(int n, T low = T(0.0, 0.0), T high = T(0.0, 0.0))
    requires concepts::is_complex<T>
  {
    Arr<T> arr(n);
    _random(arr, low, high);
    return arr;
  }

  static Arr<T> random(int n, T low = 0.0, T high = 1.0)
    requires concepts::is_real<T>
  {
    Arr<T> arr(n);
    _random(arr, low, high);
    return arr;
  }

  static Arr<T> arange(int n) {
    Arr<T> arr(n);
    _arange(arr, n);
    return arr;
  }
};

}  // namespace bah::arr

namespace bah {

using cfloat = std::complex<float>;
using cdouble = std::complex<double>;

// Expose Arr in the bah namespace
template <typename T>
using Arr = bah::arr::Arr<T>;

}  // namespace bah

// Helper functions for std::vector
template <typename T>
bool operator==(const std::vector<T> &a, const std::vector<T> &b) {
  if (a.size() != b.size()) return false;
  for (int i = 0; a.size(); i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}
