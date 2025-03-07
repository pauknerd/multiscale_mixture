#pragma once

#include <deal.II/base/tensor.h>

namespace Mixture::GrowthStrategies
{
  template <int dim, typename Number = double>
  class GrowthStrategyBase
  {
  public:
    virtual ~GrowthStrategyBase() = default;

    virtual dealii::Tensor<2, dim, Number>
    EvaluateInverseGrowthDeformationGradient(
      const Number              current_mass_fraction_ratio,
      const dealii::Point<dim> &p = dealii::Point<dim>()) const = 0;

    virtual dealii::SymmetricTensor<2, dim, Number>
    get_volumetric_stress_contribution(
      const Number                          current_mass_fraction_ratio,
      const dealii::Tensor<2, dim, Number> &F_mixture) const = 0;

    virtual dealii::SymmetricTensor<4, dim, Number>
    get_volumetric_tangent_contribution(
      const Number                          current_mass_fraction_ratio,
      const dealii::Tensor<2, dim, Number> &F_mixture) const = 0;
  };
} // namespace Mixture::GrowthStrategies
