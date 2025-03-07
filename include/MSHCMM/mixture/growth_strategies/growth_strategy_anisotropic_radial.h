#pragma once

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include "growth_strategy_base.h"

namespace Mixture::GrowthStrategies
{
  /**
   * Growth strategy for radial direction. Assumes that the cylinder axis is
   * along the z-direction.
   *
   * @tparam dim
   * @tparam Number
   */
  template <int dim, typename Number = double>
  class GrowthStrategyAnisotropicRadial : public GrowthStrategyBase<dim, Number>
  {
  public:
    GrowthStrategyAnisotropicRadial() = default;

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

      // growth direction in radial direction
      dealii::Tensor<1, dim> growth_direction =
        p - dealii::Tensor<1, dim>({0.0, 0.0, p[2]});
      growth_direction /= growth_direction.norm();
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
  };
} // namespace Mixture::GrowthStrategies
