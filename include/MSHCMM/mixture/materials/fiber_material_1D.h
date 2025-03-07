#pragma once

#include "fiber_material_base.h"

namespace Materials
{

  // active material parameters
  // todo: move to separate file
  template <typename Number>
  struct ActiveMaterialSettings
  {
    ActiveMaterialSettings(const Number sigma_act,
                           const Number initial_density,
                           const Number lambda_0,
                           const Number lambda_max)
      : sigma_act(sigma_act)
      , initial_density(initial_density)
      , lambda_0(lambda_0)
      , lambda_max(lambda_max)
    {}

    // maximal active Cauchy stress
    Number sigma_act;
    // initial reference density (constant)
    Number initial_density;
    // active stretch at zero and maximum active stress
    Number lambda_0;
    Number lambda_max;

    // active stretch in fiber direction
    // this should be set to 1.0 and doesn't change during the simulation
    const Number lambda_act{1.0};

    Number
    get_dPsi(const Number contractility) const
    {
      return contractility * sigma_act / initial_density *
             (1.0 -
              std::pow((lambda_max - lambda_act) / (lambda_max - lambda_0),
                       2.0));
    }
  };

  // todo: explicitly state which strain energy function is used
  template <int dim, typename Number>
  class FiberMaterial_1D : public FiberMaterialBase<dim, Number>
  {
  public:
    FiberMaterial_1D(const Number                  c_1,
                     const Number                  c_2,
                     const Tensor<1, dim, Number> &fiber_orientation,
                     const std::optional<ActiveMaterialSettings<Number>>
                       active_material_settings = std::nullopt)
      : c_1(c_1)
      , c_2(c_2)
      , fiber_orientation(1.0 / fiber_orientation.norm() * fiber_orientation)
      , active_material_settings(active_material_settings)
    {
      // setup fiber direction tensors
      structural_tensor =
        symmetrize(outer_product(fiber_orientation, fiber_orientation));
      orthogonal_tensor =
        Physics::Elasticity::StandardTensors<dim>::I - structural_tensor;
    };

    void
    update_material_data(const Tensor<2, dim, Number> &F,
                         const Tensor<2, dim, Number> &F_inelastic_inverse,
                         const Number                  contractility) override
    {
      // compute right Cauchy Green
      const auto C = Physics::Elasticity::Kinematics::C(F);
      // compute elastic part
      const auto C_elastic =
        symmetrize(transpose(F_inelastic_inverse) * C * F_inelastic_inverse);

      // Evaluate the first/second derivative of the free-energy function w.r.t.
      // C
      const auto dPsi_dC = evaluate_dPsidC(C_elastic, F_inelastic_inverse);
      const auto ddPsi_dCdC =
        evaluate_ddPsidCdC(C_elastic, F_inelastic_inverse);

      // Add stress contributions
      PK2_stress     = 2.0 * dPsi_dC;
      tangent_matrix = 4.0 * ddPsi_dCdC;

      // add active parts if necessary
      if (active_material_settings.has_value())
        {
          PK2_stress += evaluate_active_stress(C, contractility);
          tangent_matrix += evaluate_active_tangent(C, contractility);
        }

      // update current fiber Cauchy stress
      fiber_cauchy_stress =
        evaluate_fiber_cauchy_stress(C_elastic, contractility);
    }

    // return fiber Cauchy stress? Just a single number since 1D fiber?
    Number
    get_fiber_cauchy_stress() const override
    {
      return fiber_cauchy_stress;
    }

    // return active fiber Cauchy stress? Just a single number since 1D fiber?
    Number
    get_active_fiber_cauchy_stress() const override
    {
      return active_fiber_cauchy_stress;
    }

    // evaluate fiber Cauchy stress
    Number
    evaluate_fiber_cauchy_stress(
      const SymmetricTensor<2, dim, Number> &C_elastic,
      const Number                           contractility) const override
    {
      // see Holzapfel eq. 6.209 on page 269
      // Evaluate derivatives of Free-Energy function w.r.t. to pseudo invariant
      const double I_4 = fiber_orientation * C_elastic * fiber_orientation;
      // Strain energy function based on Latorre, note the factor 0.5
      const auto dPsi =
        c_1 * (I_4 - 1.0) * std::exp(c_2 * (I_4 - 1.0) * (I_4 - 1.0)) * 0.5;

      // get active contribution if necessary
      const auto dPsi_active =
        active_material_settings.has_value() ?
          active_material_settings->get_dPsi(contractility) :
          0.0;

      // compute J_e, that is sqrt of determinant of C_elastic
      const double J_e = std::sqrt(dealii::determinant(C_elastic));

      // store active stress for output
      active_fiber_cauchy_stress = 1.0 / J_e * dPsi_active;

      // store total fiber stress
      fiber_cauchy_stress = 1.0 / J_e * (2.0 * dPsi * I_4 + dPsi_active);

      return fiber_cauchy_stress;
    }

    const SymmetricTensor<2, dim, Number> &
    get_structural_tensor() const override
    {
      return structural_tensor;
    }

    const SymmetricTensor<2, dim, Number> &
    get_orthogonal_structural_tensor() const override
    {
      return orthogonal_tensor;
    }

    const Tensor<1, dim, Number> &
    get_fiber_direction() const override
    {
      return fiber_orientation;
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

    SymmetricTensor<2, dim, Number>
    get_dsig_dCe(
      const SymmetricTensor<2, dim, Number> &C_elastic) const override
    {
      // Evaluate derivatives of Free-Energy function w.r.t. to pseudo invariant
      const double I_4 = fiber_orientation * C_elastic * fiber_orientation;
      // first derivative
      const auto dPI = c_1 * (I_4 - 1.0) *
                       std::exp(c_2 * (I_4 - 1.0) * (I_4 - 1.0)) *
                       0.5; // use * 0.5 if recreating Latorre
      // second derivative
      const auto ddPII = (1.0 + 2.0 * c_2 * (I_4 - 1.0) * (I_4 - 1.0)) * c_1 *
                         exp(c_2 * (I_4 - 1.0) * (I_4 - 1.0)) *
                         0.5; // use * 0.5 if recreating Latorre

      return 2.0 * (ddPII * I_4 + dPI) * structural_tensor;
    }

    Number
    get_dsig_dlambdae(const Number I_4) const override
    {
      // based on Maes et al., 2023
      // compute dW/dI_4
      const auto dW_dI_4 = c_1 * (I_4 - 1.0) *
                           std::exp(c_2 * (I_4 - 1.0) * (I_4 - 1.0)) *
                           0.5; // use * 0.5 if recreating Latorre
      // compute ddW/ddI_4
      const auto ddW_ddI_4 = (1.0 + 2.0 * c_2 * (I_4 - 1.0) * (I_4 - 1.0)) *
                             c_1 * exp(c_2 * (I_4 - 1.0) * (I_4 - 1.0)) *
                             0.5; // use * 0.5 if recreating Latorre

      return (ddW_ddI_4 * I_4 * I_4 + dW_dI_4 * I_4);
    }

  private:
    SymmetricTensor<2, dim, Number>
    evaluate_dPsidC(const SymmetricTensor<2, dim, Number> &C_elastic,
                    const Tensor<2, dim, Number> &F_inelastic_inverse) const
    {
      // Evaluate derivatives of Free-Energy function w.r.t. to pseudo invariant
      const double I_4 = fiber_orientation * C_elastic * fiber_orientation;
      const auto   dPI_aniso = c_1 * (I_4 - 1.0) *
                             std::exp(c_2 * (I_4 - 1.0) * (I_4 - 1.0)) *
                             0.5; // use * 0.5 if recreating Latorre

      // Return derivative
      const SymmetricTensor<2, dim, Number> product = symmetrize(
        F_inelastic_inverse * structural_tensor * F_inelastic_inverse);

      return dPI_aniso * product;
    }

    SymmetricTensor<4, dim, Number>
    evaluate_ddPsidCdC(const SymmetricTensor<2, dim, Number> &C_elastic,
                       const Tensor<2, dim, Number> &F_inelastic_inverse) const
    {
      // compute I_4
      const double I_4 = fiber_orientation * C_elastic * fiber_orientation;

      // Compute inverse inelastic right Cauchy-Green deformation tensor
      const auto C_inelastic_inverse =
        symmetrize(F_inelastic_inverse * transpose(F_inelastic_inverse));

      // Evaluate second derivative of the free-energy function with respect to
      // the anisotropic invariants
      const double ddPsiIIe_aniso =
        (1.0 + 2.0 * c_2 * std::pow((I_4 - 1.0), 2)) * c_1 *
        std::exp(c_2 * std::pow((I_4 - 1.0), 2)) *
        0.5; // use * 0.5 if recreating Latorre

      return ddPsiIIe_aniso *
             std::pow(scalar_product(C_inelastic_inverse, structural_tensor),
                      2.0) *
             outer_product(structural_tensor, structural_tensor);
    }

    SymmetricTensor<2, dim, Number>
    evaluate_active_stress(const SymmetricTensor<2, dim, Number> &C,
                           const Number contractility) const
    {
      // Evaluate derivatives of Free-Energy function w.r.t. to pseudo invariant
      // corresponds to lambda^2
      const Number I_4      = fiber_orientation * C * fiber_orientation;
      const auto dPI_active = active_material_settings->get_dPsi(contractility);

      return dPI_active * 1.0 / I_4 * structural_tensor;
    }

    SymmetricTensor<4, dim, Number>
    evaluate_active_tangent(const SymmetricTensor<2, dim, Number> &C,
                            const Number contractility) const
    {
      // Evaluate derivatives of Free-Energy function w.r.t. to pseudo invariant
      // corresponds to lambda^2
      const double I_4      = fiber_orientation * C * fiber_orientation;
      const auto dPI_active = active_material_settings->get_dPsi(contractility);

      return -2.0 * dPI_active * std::pow(1.0 / I_4, 2.0) *
             outer_product(structural_tensor, structural_tensor);
    }

    // material parameters
    Number c_1;
    Number c_2;

    // fiber orientation a_0 in reference configuration
    Tensor<1, dim, Number> fiber_orientation;
    // a_0 x a_0
    SymmetricTensor<2, dim, Number> structural_tensor;
    // I - a_0 x a_0
    SymmetricTensor<2, dim, Number> orthogonal_tensor;

    // active material settings, optional
    std::optional<ActiveMaterialSettings<Number>> active_material_settings;

    // fiber Cauchy stress
    mutable Number fiber_cauchy_stress{0};
    mutable Number active_fiber_cauchy_stress{0};

    // constituent-level PK2 stress and tangent matrix
    SymmetricTensor<2, dim, Number> PK2_stress{
      SymmetricTensor<2, dim, Number>()};
    SymmetricTensor<4, dim, Number> tangent_matrix{
      SymmetricTensor<4, dim, Number>()};
  };
} // namespace Materials
