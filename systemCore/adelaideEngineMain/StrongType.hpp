#pragma once
#include <iostream>
#include <type_traits>

namespace qip {

template <auto enumV, typename BaseT> struct StrongType {
private:
  static_assert(std::is_arithmetic_v<BaseT>,
                "StrongType only available for arithmetic types");
  static_assert(
      std::is_enum_v<decltype(enumV)>,
      "StrongType must be instantiated with scoped enum (enum class)");
  using StrongT = StrongType<enumV, BaseT>; // type alias

public:
  BaseT value;

  explicit constexpr StrongType(BaseT tv) : value(tv) {}
  explicit constexpr operator BaseT() const { return value; }
  constexpr BaseT &as_base() { return value; }
  [[nodiscard]] constexpr BaseT as_base() const { return value; }

  using BaseType = BaseT; // makes 'BaseType' publicly accessible

  //! Provides operators for regular arithmetic operations
  constexpr StrongT &operator*=(const StrongT &rhs) {
    this->value *= rhs.value;
    return *this;
  }
  friend constexpr StrongT operator*(StrongT lhs, const StrongT &rhs) {
    return lhs *= rhs;
  }
  constexpr StrongT &operator/=(const StrongT &rhs) {
    this->value /= rhs.value;
    return *this;
  }
  friend constexpr StrongT operator/(StrongT lhs, const StrongT &rhs) {
    return lhs /= rhs;
  }
  constexpr StrongT &operator+=(const StrongT &rhs) {
    this->value += rhs.value;
    return *this;
  }
  friend constexpr StrongT operator+(StrongT lhs, const StrongT &rhs) {
    return lhs += rhs;
  }
  constexpr StrongT &operator-=(const StrongT &rhs) {
    this->value -= rhs.value;
    return *this;
  }
  friend constexpr StrongT operator-(StrongT lhs, const StrongT &rhs) {
    return lhs -= rhs;
  }

  //! Provide Base*Strong, Strong*Base oprators - allow scalar multiplication
  constexpr StrongT &operator*=(const BaseT &rhs) {
    this->value *= rhs;
    return *this;
  }
  friend constexpr StrongT operator*(StrongT lhs, const BaseT &rhs) {
    return lhs *= rhs;
  }
  friend constexpr StrongT operator*(const BaseT &lhs, StrongT rhs) {
    return rhs *= lhs;
  }
  //! Provide Strong/Base, but NOT Base/Strong (still scalar multiplication).
  // If StrongT is used for physical units, this will likely not be what you
  // want. In this case, just be explicit. Base/Strong is not scalar
  // multiplication.
  constexpr StrongT &operator/=(const BaseT &rhs) {
    this->value /= rhs;
    return *this;
  }
  friend constexpr StrongT operator/(StrongT lhs, const BaseT &rhs) {
    return lhs /= rhs;
  }

  //! Provides pre/post increment/decrement (++, --) operators
  constexpr StrongT &operator++() {
    ++value;
    return *this;
  }
  constexpr StrongT operator++(int) {
    StrongT result(*this);
    ++(*this);
    return result;
  }
  constexpr StrongT &operator--() {
    --value;
    return *this;
  }
  constexpr StrongT operator--(int) {
    StrongT result(*this);
    --(*this);
    return result;
  }

  //! Provides comparison operators
  friend constexpr bool operator==(const StrongT &lhs, const StrongT &rhs) {
    return lhs.value == rhs.value;
  }
  friend constexpr bool operator!=(const StrongT &lhs, const StrongT &rhs) {
    return !(lhs == rhs);
  }
  friend constexpr bool operator<(const StrongT &lhs, const StrongT &rhs) {
    return lhs.value < rhs.value;
  }
  friend constexpr bool operator>(const StrongT &lhs, const StrongT &rhs) {
    return rhs < lhs;
  }
  friend constexpr bool operator<=(const StrongT &lhs, const StrongT &rhs) {
    return !(rhs < lhs);
  }
  friend constexpr bool operator>=(const StrongT &lhs, const StrongT &rhs) {
    return !(lhs < rhs);
  }

  //! Provides operators for direct comparison w/ BaseT literal (rvalue).
  //! Note: Does not allow comparison with BaseT lvalue
  friend constexpr bool operator==(const StrongT &lhs, const BaseT &&rhs) {
    return lhs.value == rhs;
  }
  friend constexpr bool operator!=(const StrongT &lhs, const BaseT &&rhs) {
    return lhs.value != rhs;
  }
  friend constexpr bool operator<(const StrongT &lhs, const BaseT &&rhs) {
    return lhs.value < rhs;
  }
  friend constexpr bool operator>(const StrongT &lhs, const BaseT &&rhs) {
    return lhs.value > rhs;
  }
  friend constexpr bool operator<=(const StrongT &lhs, const BaseT &&rhs) {
    return lhs.value <= rhs;
  }
  friend constexpr bool operator>=(const StrongT &lhs, const BaseT &&rhs) {
    return lhs.value >= rhs;
  }
  friend constexpr bool operator==(const BaseT &&lhs, const StrongT &rhs) {
    return lhs == rhs.value;
  }
  friend constexpr bool operator!=(const BaseT &&lhs, const StrongT &rhs) {
    return lhs != rhs.value;
  }
  friend constexpr bool operator<(const BaseT &&lhs, const StrongT &rhs) {
    return lhs < rhs.value;
  }
  friend constexpr bool operator>(const BaseT &&lhs, const StrongT &rhs) {
    return lhs > rhs.value;
  }
  friend constexpr bool operator<=(const BaseT &&lhs, const StrongT &rhs) {
    return lhs <= rhs.value;
  }
  friend constexpr bool operator>=(const BaseT &&lhs, const StrongT &rhs) {
    return lhs >= rhs.value;
  }

  //! Provides iostream interface, works as it would for BaseT
  friend std::ostream &operator<<(std::ostream &os, const StrongT &rhs) {
    return os << rhs.value;
  }
  friend std::istream &operator>>(std::istream &is, StrongT &rhs) {
    return is >> rhs.value;
  }
};

} // namespace qip
