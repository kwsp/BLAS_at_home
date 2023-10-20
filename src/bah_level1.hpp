#pragma once

namespace bah {

/**
 * cblas_?asum
 * https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/cblas-asum.html
 */
float cblas_sasum(const int n, const float *x, const int incX);
float cblas_scasum(const int n, const void *x, const int incx);
double cblas_dasum(const int n, const double *x, const int incx);
double cblas_dzasum(const int n, const void *x, const int incx);

}  // namespace bah
