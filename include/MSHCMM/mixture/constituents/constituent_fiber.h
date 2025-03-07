#pragma once

#include "deal.II/base/utilities.h"

#include <deal.II/sundials/arkode.h>

#include <memory>

#include "arkode/arkode_arkstep.h"
#include "constituent_base.h"

namespace Mixture::Constituents
{

  /**
   * Implementation of fiber constituent.
   *
   * @tparam dim
   * @tparam Number
   */
  template <int dim, typename Number = double>
  class ConstituentFiber : public ConstituentBase<dim, Number>
  {
  public:
    explicit ConstituentFiber(
      unsigned int                   constituent_id,
      Number                         prestretch,
      dealii::Function<dim, Number> &prestretch_function,
      std::unique_ptr<Materials::FiberMaterialBase<dim, Number>> material,
      Number                                                     time_step,
      Number                                                     decay_time,
      Number                                                     gain);

    //! @bried Update all material quantities, called every Newton iteration.
    void
    update_material(const dealii::Tensor<2, dim, Number> &F,
                    const dealii::Tensor<2, dim, Number> &F_g_inv) override;

    //! @brief Update remodeling stretch and growth term, called once AFTER Newton solve was
    // successfull, i.e., at the end of a time step.
    void
    update_GR(const dealii::Tensor<2, dim, Number> &F,
              const dealii::Tensor<2, dim, Number> &F_g_inv,
              Number initial_mass_fraction) override;

    // evaluate prestretch function at current time (set previously) and given
    // location p
    void
    evaluate_prestretch(const dealii::Point<dim> &p) override;

    const dealii::SymmetricTensor<2, dim, Number> &
    get_stress() const override;

    const dealii::SymmetricTensor<4, dim, Number> &
    get_tangent() const override;

    const dealii::Tensor<2, dim, Number> &
    get_prestretch_tensor() const override;

    dealii::Tensor<2, dim, Number> &
    get_prestretch_tensor() override;

    Number
    get_mass_fraction_ratio() const override;

    TransferableParameters<Number> &
    get_transferable_parameters() override;

    const dealii::Tensor<1, dim, Number> &
    get_fiber_direction() const override;

    [[nodiscard]] const std::string &
    get_name() const override;

    [[nodiscard]] unsigned int
    get_constituent_id() const override;

  private:
    void
    update_growth_and_remodeling(const dealii::Tensor<2, dim, Number> &F,
                                 const dealii::Tensor<2, dim, Number> &F_gr_inv,
                                 Number initial_mass_fraction);

    void
    update_remodeling(const dealii::Tensor<2, dim, Number> &F,
                      const dealii::Tensor<2, dim, Number> &F_g_inv,
                      const Number                          delta_sigma,
                      const bool                            coupled);

    // Update growth. If initial_mass_fraction is < 0.0, switch to classical
    // model. This is automatically done if the mixture problem is setup with
    // coupled = false.
    void
    update_growth(Number delta_sigma, Number initial_mass_fraction);

    void
    update_F_r_inv();

    void
    update_transferable_parameters();

    void
    update_sigma_h();

    dealii::Tensor<2, dim, Number>
    create_prestretch_tensor();

    Number
    evaluate_lambda_r_dot(const dealii::Tensor<2, dim, Number> &F,
                          const dealii::Tensor<2, dim, Number> &F_g_inv,
                          const Number                          lambda_r,
                          const Number                          delta_sigma,
                          const bool                            coupled) const
    {
      // compute inverse of remodeling deformation gradient
      const auto F_r_inv_temp =
        lambda_pre / lambda_r * material->get_structural_tensor() +
        std::sqrt(lambda_r / lambda_pre) *
          material->get_orthogonal_structural_tensor();
      // compute F_gr_inv, total inelastic deformation
      const auto F_gr_inv = F_g_inv * F_r_inv_temp;
      // compute right Cauchy Green
      const auto C = dealii::Physics::Elasticity::Kinematics::C(F);
      // compute elastic part
      const auto C_elastic =
        dealii::symmetrize(transpose(F_gr_inv) * C * F_gr_inv);
      // compute I_4 (invariant)
      const auto I_4 = C_elastic * material->get_structural_tensor();
      // get term in brackets
      const auto temp = material->get_dsig_dlambdae(I_4);
      // determinant of F
      const auto J = dealii::determinant(F);

      // this should not contain the prefactor J_g/J since this is already
      // included
      const Number delta_sigma_growth =
        material->get_fiber_cauchy_stress() - sig_h;

      // based on Maes et al., 2023
      if (coupled)
        return (transferable_parameters.mass_production /
                transferable_parameters.current_mass_fraction) *
               lambda_r * delta_sigma * J / (4.0 * temp);
      else
        return (k_sig * delta_sigma_growth / sig_h +
                1.0 / transferable_parameters.decay_time) *
               lambda_r * delta_sigma * J / (4.0 * temp);
    }

    // name
    std::string name{"fiber"};
    // constituent id
    unsigned int constituent_id;
    // time step
    Number dt{1.0};
    // transferable parameters
    TransferableParameters<Number> transferable_parameters;
    // mass fraction ratio, ratio of current mass fraction to initial mass
    // fraction todo: should this be part of the transferable_parameters?
    Number mass_fraction_ratio{1.0};

    // prestretch
    Number                         lambda_pre{1.0};
    dealii::Tensor<2, dim, Number> prestretch_tensor;
    // prestretch function
    dealii::Function<dim, Number> &prestretch_function;

    // remodeling stretch
    Number current_lambda_r{1.0};

    // inverse remodeling deformation gradient
    dealii::SymmetricTensor<2, dim, Number> F_r_inv{
      dealii::Physics::Elasticity::StandardTensors<dim>::I};
    // growth gain factor
    Number k_sig;
    // sigma_h
    Number sig_h{0.0};
    // material
    std::unique_ptr<Materials::FiberMaterialBase<dim, Number>> material;
  };
} // namespace Mixture::Constituents
