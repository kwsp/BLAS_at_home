#include "bah_level1.hpp"

#include <cmath>

// BLAS at home
namespace bah {

float cblas_sasum(const int n, const float *x, const int incx) {
  float sum{0.0f};
  for (int i = 0; i < n; i += incx) sum += std::fabs(x[i]);
  return sum;
}

float cblas_scasum(const int n, const void *x, const int incx) {
  // TODO
  return 0.0;
}

double cblas_dasum(const int n, const double *x, const int incx) {
  double sum{0.0};
  for (int i = 0; i < n; i += incx) sum += std::abs(x[i]);
  return 0.0;
}

double cblas_dzasum(const int n, const void *x, const int incx) {
  // TODO
  return 0.0;
}

}  // namespace bah
