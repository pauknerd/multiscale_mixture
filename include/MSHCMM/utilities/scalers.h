#pragma once

template <typename Number = double>
using Scaler = std::function<Number(const Number &input)>;

template <typename Number = double>
using ScalerList = std::vector<Scaler<Number>>;

using SourceIndices   = std::vector<unsigned int>;
using DestinationPair = std::pair<unsigned int, unsigned int>;

namespace Scalers
{
  // transform an input from the range [0, 1] to the real values
  template <typename Number = double>
  inline Scaler<Number>
  log(const Number a = 1.0, const Number b = 1.0, const Number c = 1.0)
  {
    return [a, b, c](const Number &x) -> Number {
      return a * std::log(b / (1.0 - c * x));
    };
  }

  // transform a real-valued input into the range [0, 1]
  template <typename Number = double>
  inline Scaler<Number>
  exponential(const Number sensitivity     = 1.0,
              const Number half_activation = 0.0,
              const Number amplitude       = 1.0)
  {
    return
      [sensitivity, half_activation, amplitude](const Number &x) -> Number {
        return amplitude * std::exp(sensitivity * (x - half_activation)) /
               (1.0 + std::exp(sensitivity * (x - half_activation)));
      };
  }

  // transform a real-valued input into the range [0, 1]
  template <typename Number = double>
  inline Scaler<Number>
  NEW_exponential(const Number sensitivity = 1.0, const Number y_0 = 0.5)
  {
    Assert(y_0 > 0.0, dealii::ExcMessage("y_0 must be in interval (0, 1)!"));
    Assert(y_0 < 1.0, dealii::ExcMessage("y_0 must be in interval (0, 1)!"));
    // compute shift value
    const Number a = std::log(1.0 / y_0 - 1.0);

    return [sensitivity, a](const Number &x) -> Number {
      return 1.0 / (1.0 + std::exp(a - sensitivity * x));
    };
  }

  // identity
  template <typename Number = double>
  inline Scaler<Number>
  identity()
  {
    return [](const Number &x) -> Number { return x; };
  }

  template <typename Number = double>
  inline Scaler<Number>
  linear(const Number slope       = 1.0,
         const Number shift       = 0.0,
         const Number y_intercept = 0.0)
  {
    return [slope, shift, y_intercept](const Number &x) -> Number {
      return slope * (x - shift) + y_intercept;
    };
  }
} // namespace Scalers
