#pragma once

#include <complex>
#include <random>
#include <type_traits>
#include <vector>

namespace bah::array {

// Type traits for std::complex
template <typename T>
struct is_complex {
  static constexpr bool value = false;
};

template <template <typename...> class C, typename U>
struct is_complex<C<U>> {
  static constexpr bool value = std::is_same<C<U>, std::complex<U>>::value;
};

// Helper functions for std::vector

template <typename Tp>
bool operator==(const std::vector<Tp> &a, const std::vector<Tp> &b) {
  if (a.size() != b.size()) return false;
  for (int i = 0; a.size(); i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

template <typename RealTp>
std::enable_if_t<!is_complex<RealTp>::value, void> random_(
    std::vector<RealTp> &arr, RealTp low, RealTp high) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(low, high);
  for (int i = 0; i < arr.size(); i++) {
    arr[i] = dis(gen);
  }
}

template <typename CxTp>
std::enable_if_t<is_complex<CxTp>::value, void> random_(std::vector<CxTp> &arr,
                                                        CxTp low, CxTp high) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis_real(low.real(), high.real());
  std::uniform_real_distribution<> dis_imag(low.imag(), high.imag());
  for (int i = 0; i < arr.size(); i++) {
    arr[i] = CxTp(dis_real(gen), dis_imag(gen));
  }
}

template <typename RealTp = double>
std::enable_if_t<!is_complex<RealTp>::value, std::vector<RealTp>> random(
    size_t N, RealTp low = 0.0, RealTp high = 1.0) {
  std::vector<RealTp> arr(N);
  random_(arr, low, high);
  return arr;
}

// TODO finish random gen for complex numbers
// Then test the L1 functions for complex
template <typename CxTp>
std::enable_if_t<is_complex<CxTp>::value, std::vector<CxTp>> random(
    size_t N, CxTp low = CxTp(0.0, 0.0), CxTp high = CxTp(1.0, 1.0)) {
  std::vector<CxTp> arr(N);
  random_(arr, low, high);
  return arr;
}

template <typename RealTp>
void arange_(std::vector<RealTp> &arr, size_t n) {
  assert(arr.size() >= n);
  for (int i = 0; i < n; i++) arr[i] = static_cast<RealTp>(i);
}

template <typename RealTp>
auto arange(size_t n) {
  std::vector<RealTp> arr(n);
  arange_(arr, n);
  return arr;
}

}  // namespace bah::array
