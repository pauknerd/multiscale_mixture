#pragma once

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include "growth_strategy_base.h"

namespace Mixture::GrowthStrategies
{
  template <int dim, typename Number = double>
  class GrowthStrategyAnisotropic : public GrowthStrategyBase<dim, Number>
  {
  public:
    explicit GrowthStrategyAnisotropic(const dealii::Tensor<1, dim> &direction)
      : growth_direction(direction)
    {}

    // evaluate the inverse growth deformation gradients
    dealii::Tensor<2, dim, Number>
    EvaluateInverseGrowthDeformationGradient(
      const Number                      current_mass_fraction_ratio,
      const dealii::Point<dim, Number> &p =
        dealii::Point<dim, Number>()) const override
    {
      (void)p;
      // create identity tensor
      const auto identity =
        dealii::Physics::Elasticity::StandardTensors<dim>::I;
      dealii::Tensor<2, dim, Number> inverse_growth_def_grad;

      // create structural tensor based on growth direction
      const auto structural_tensor =
        symmetrize(outer_product(growth_direction, growth_direction));

      inverse_growth_def_grad =
        (1. / current_mass_fraction_ratio - 1.0) * structural_tensor + identity;

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

  private:
    dealii::Tensor<1, dim> growth_direction;
  };
} // namespace Mixture::GrowthStrategies
