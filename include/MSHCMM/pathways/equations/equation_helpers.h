#pragma once

namespace
{
  namespace EquationHelpers
  {
    inline const auto B = [](const double EC_50, const double n) {
      return (std::pow(EC_50, n) - 1) / (2 * std::pow(EC_50, n) - 1);
    };

    inline const auto K = [](const double B, const double n) {
      return std::pow(B - 1, 1 / n);
    };


    // define lambda for f_act
    inline const auto act =
      [](const double X, const double w, const double n, const double EC_50) {
        const double value = w * (B(EC_50, n) * std::pow(X, n)) /
                             (std::pow(K(B(EC_50, n), n), n) + std::pow(X, n));
        return value <= 1 ? value : 1;
      };


    // define lambda for f_inhib
    inline const auto inhib =
      [](const double X, const double w, const double n, const double EC_50) {
        return (1 - act(X, w, n, EC_50));
      };

    // AND function for an arbitrary number of inputs
    template <typename... Args>
    inline double
    AND(Args... args)
    {
      return (... * args);
    }

    // OR function for TWO inputs
    inline double
    OR(double f1, double f2)
    {
      return f1 + f2 - f1 * f2;
    }
  } // namespace EquationHelpers
} // namespace
