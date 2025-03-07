#pragma once

#include <deal.II/physics/elasticity/standard_tensors.h>

#include "growth_strategy_base.h"

namespace Mixture::GrowthStrategies
{
  template <int dim, typename Number = double>
  class GrowthStrategyIsotropic : public GrowthStrategyBase<dim, Number>
  {
  public:
    GrowthStrategyIsotropic() = default;

    // evaluate the inverse growth deformation gradients
    dealii::Tensor<2, dim, Number>
    EvaluateInverseGrowthDeformationGradient(
      const Number                      current_mass_fraction_ratio,
      const dealii::Point<dim, Number> &p =
        dealii::Point<dim, Number>()) const override
    {
      // create identity tensor
      const auto identity =
        dealii::Physics::Elasticity::StandardTensors<dim>::I;
      dealii::Tensor<2, dim, Number> inverse_growth_def_grad;

      inverse_growth_def_grad =
        std::pow(current_mass_fraction_ratio, -1.0 / 3.0) * identity;

      return inverse_growth_def_grad;
    }

    dealii::SymmetricTensor<2, dim, Number>
    get_volumetric_stress_contribution(
      const Number                          current_mass_fraction_ratio,
      const dealii::Tensor<2, dim, Number> &F_mixture) const override
    {
      (void)current_mass_fraction_ratio;
      (void)F_mixture;
      // return 0 tensor
      return dealii::SymmetricTensor<2, dim, Number>();
    }

    dealii::SymmetricTensor<4, dim, Number>
    get_volumetric_tangent_contribution(
      const Number                          current_mass_fraction_ratio,
      const dealii::Tensor<2, dim, Number> &F_mixture) const override
    {
      (void)current_mass_fraction_ratio;
      (void)F_mixture;
      // return 0 tensor
      return dealii::SymmetricTensor<4, dim, Number>();
    }
  };
} // namespace Mixture::GrowthStrategies
