#include <random>
#include <vector>

namespace bah::array {

template <typename RealTp>
void random_(std::vector<RealTp> &arr, RealTp low = 0.0, RealTp high = 1.0) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(low, high);
  for (int i = 0; i < arr.size(); i++) arr[i] = dis(gen);
}

template <typename Tp>
bool operator==(const std::vector<Tp> &a, const std::vector<Tp> &b) {
  if (a.size() != b.size()) return false;
  for (int i = 0; a.size(); i++)
    if (a[i] != b[i]) return false;
  return true;
}

template <typename RealTp = double>
auto random(size_t N, RealTp low = 0.0, RealTp high = 1.0) {
  std::vector<RealTp> arr(N);
  random_(arr, low, high);
  return arr;
}

}  // namespace bah::array
