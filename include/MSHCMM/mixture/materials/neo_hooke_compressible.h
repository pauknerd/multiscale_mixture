#pragma once

#include "material_base.h"

namespace Materials
{
  template <int dim, typename Number>
  class NeoHooke_compressible : public MaterialBase<dim, Number>
  {
  public:
    NeoHooke_compressible(const Number mu, const Number K)
      : mu(mu)
      , K(K)
    {}

    // update material data at quadrature points
    void
    update_material_data(
      const Tensor<2, dim, Number> &F,
      const Tensor<2, dim, Number> &F_inelastic_inverse) override
    {
      const auto F_elastic = F * F_inelastic_inverse;
      // compute right Cauchy-Green
      const auto C         = Physics::Elasticity::Kinematics::C(F);
      const auto C_inverse = invert(C);

      // compute inelastic right Cauchy-Green
      const auto C_inelastic_inverse =
        symmetrize(F_inelastic_inverse * transpose(F_inelastic_inverse));

      // compute C_elastic
      const auto C_elastic =
        symmetrize(transpose(F_inelastic_inverse) * C * F_inelastic_inverse);

      // compute invariants of C_elastic
      const auto I_1 = first_invariant(C_elastic);
      const auto I_3 = third_invariant(C_elastic);

      // second Piola Kirchhoff stress
      const double gamma_1 = mu * std::pow(I_3, -1. / 3.);
      const double gamma_3 = -1. / 3. * mu * I_1 * std::pow(I_3, -1. / 3.);
      const double J_e     = determinant(F_elastic);
      // pressure from constitutive equation 0.5 * K * (J_e - 1.0)^2
      const auto p              = K * (J_e - 1.0);
      const auto PK2_stress_vol = J_e * p * C_inverse;

      PK2_stress =
        gamma_1 * C_inelastic_inverse + gamma_3 * C_inverse + PK2_stress_vol;

      // compute delta factors for tangent
      const double delta_3 = -2. / 3. * mu * std::pow(I_3, -1. / 3.);
      const double delta_6 =
        2. / 3. * mu * std::pow(I_3, -1. / 3.) * 1. / 3. * I_1;
      const double delta_7 = 2. / 3. * mu * std::pow(I_3, -1. / 3.) * I_1;

      // volumetric contribution
      // note: recall that dC_inv_dC(F) returns -0.5 * (...)
      auto tangent_matrix_vol =
        (J_e * p + J_e * J_e * K) *
          dealii::outer_product(C_inverse, C_inverse) +
        2.0 * J_e * p *
          (dealii::Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(F));

      tangent_matrix =
        delta_3 * (outer_product(C_inelastic_inverse, C_inverse) +
                   outer_product(C_inverse, C_inelastic_inverse)) +
        delta_6 * outer_product(C_inverse, C_inverse) +
        delta_7 * -Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(F) +
        tangent_matrix_vol;
    }

    const SymmetricTensor<2, dim, Number> &
    get_stress() const override
    {
      return PK2_stress;
    }

    const SymmetricTensor<4, dim, Number> &
    get_tangent() const override
    {
      return tangent_matrix;
    }

  private:
    // material parameter, shear modulus
    Number mu;
    Number K;

    // constituent-level PK2 stress and tangent matrix
    SymmetricTensor<2, dim, Number> PK2_stress{
      SymmetricTensor<2, dim, Number>()};
    SymmetricTensor<4, dim, Number> tangent_matrix{
      SymmetricTensor<4, dim, Number>()};
  };
} // namespace Materials
