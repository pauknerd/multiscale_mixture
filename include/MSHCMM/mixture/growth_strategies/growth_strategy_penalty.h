#pragma once

#include <deal.II/physics/elasticity/standard_tensors.h>

#include "growth_strategy_base.h"

namespace Mixture::GrowthStrategies
{

  template <int dim, typename Number = double>
  class GrowthStrategyPenalty : public GrowthStrategyBase<dim, Number>
  {
  public:
    explicit GrowthStrategyPenalty(const Number penalty_parameter)
      : kappa(penalty_parameter)
    {}

    // evaluate the inverse growth deformation gradients
    dealii::Tensor<2, dim, Number>
    EvaluateInverseGrowthDeformationGradient(
      const Number              current_mass_fraction_ratio,
      const dealii::Point<dim> &p = dealii::Point<dim>()) const override
    {
      (void)p;
      (void)current_mass_fraction_ratio;
      // according to Braeu et al., 2017
      return dealii::Physics::Elasticity::StandardTensors<dim>::I;
    }

    dealii::SymmetricTensor<2, dim, Number>
    get_volumetric_stress_contribution(
      const Number                          growth_factor,
      const dealii::Tensor<2, dim, Number> &F_mixture) const override
    {
      const auto det_F = determinant(F_mixture);
      const auto C     = dealii::Physics::Elasticity::Kinematics::C(F_mixture);
      const auto C_inverse = invert(C);

      return det_F * det_F * kappa * (1.0 - growth_factor / det_F) * C_inverse;
    }

    dealii::SymmetricTensor<4, dim, Number>
    get_volumetric_tangent_contribution(
      const Number                          growth_factor,
      const dealii::Tensor<2, dim, Number> &F_mixture) const override
    {
      const auto det_F = determinant(F_mixture);
      const auto C     = dealii::Physics::Elasticity::Kinematics::C(F_mixture);
      const auto C_inverse = invert(C);

      const auto I_3 = det_F * det_F;

      const double dPsi  = 0.5 * kappa * (1.0 - growth_factor / det_F);
      const double ddPsi = 0.25 * kappa * growth_factor / std::pow(det_F, 3.0);

      const double delta_6 = 4.0 * (I_3 * dPsi + I_3 * I_3 * ddPsi);
      const double delta_7 = -4.0 * I_3 * dPsi;

      return delta_6 * dealii::outer_product(C_inverse, C_inverse) +
             delta_7 *
               (-dealii::Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(
                 F_mixture));
    }

  private:
    Number kappa;
  };
} // namespace Mixture::GrowthStrategies
