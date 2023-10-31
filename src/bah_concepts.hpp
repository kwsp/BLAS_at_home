#pragma once

#include <complex>
#include <concepts>
#include <type_traits>

namespace bah::concepts {

template <typename T>
concept is_complex = std::is_same_v<T, std::complex<typename T::value_type>>;

template <typename T>
concept is_real = std::is_floating_point_v<T>;

template <typename T>
concept is_function = std::is_function_v<T>;

}  // namespace bah::concepts
