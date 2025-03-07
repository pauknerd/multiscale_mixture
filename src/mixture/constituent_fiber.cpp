#include <deal.II/sundials/arkode.h>

#include <MSHCMM/mixture/constituents/constituent_fiber.h>

#include "MSHCMM/utilities/helpers.h"

namespace Mixture::Constituents
{

  template <int dim, typename Number>
  ConstituentFiber<dim, Number>::ConstituentFiber(
    const unsigned int             constituent_id,
    const Number                   prestretch,
    dealii::Function<dim, Number> &prestretch_function,
    std::unique_ptr<Materials::FiberMaterialBase<dim, Number>> material,
    const Number                                               time_step,
    const Number                                               decay_time,
    const Number                                               gain)
    : constituent_id(constituent_id)
    , dt(time_step)
    , lambda_pre(prestretch)
    , prestretch_function(prestretch_function)
    , k_sig(gain) // todo: move to transferable parameters?
    , material(std::move(material))
  {
    // set decay time in transferable parameters
    transferable_parameters.decay_time = decay_time;

    // setup prestretch tensor
    prestretch_tensor = create_prestretch_tensor();

    // set sigma_h based on prestretch, important to create prestretch tensor
    // first todo: contains redundant calls
    update_sigma_h();
    update_transferable_parameters();
    update_F_r_inv();
  }

  template <int dim, typename Number>
  inline void
  ConstituentFiber<dim, Number>::update_material(
    const dealii::Tensor<2, dim, Number> &F,
    const dealii::Tensor<2, dim, Number> &F_g_inv)
  {
    // compute inverse of F_inelastic, i.e. F_gr_inverse
    const auto F_inelastic_inv = F_g_inv * F_r_inv;

    // update material
    material->update_material_data(F,
                                   F_inelastic_inv,
                                   transferable_parameters.contractility);

    // update fiber Cauchy stress in transferable parameters
    transferable_parameters.fiber_cauchy_stress =
      material->get_fiber_cauchy_stress();
    transferable_parameters.active_fiber_cauchy_stress =
      material->get_active_fiber_cauchy_stress();
  }

  template <int dim, typename Number>
  inline void
  ConstituentFiber<dim, Number>::update_GR(
    const dealii::Tensor<2, dim, Number> &F,
    const dealii::Tensor<2, dim, Number> &F_g_inv,
    const Number                          initial_mass_fraction)
  {
    // update growth and remodeling
    update_growth_and_remodeling(F, F_g_inv, initial_mass_fraction);
  }

  // todo: change to take a time (and position) argument? And store a (shared)
  // pointer to the prestretch function?
  template <int dim, typename Number>
  void
  ConstituentFiber<dim, Number>::evaluate_prestretch(
    const dealii::Point<dim> &p)
  {
    // update prestretch
    lambda_pre = prestretch_function.value(p);
    // create prestretch tensor for new lambda_pre
    prestretch_tensor = create_prestretch_tensor();
    // update related values, note that sigma_h depends on prestretch
    update_sigma_h();
    update_transferable_parameters();
    update_F_r_inv();
  }

  template <int dim, typename Number>
  inline const dealii::SymmetricTensor<2, dim, Number> &
  ConstituentFiber<dim, Number>::get_stress() const
  {
    return material->get_stress();
  }

  template <int dim, typename Number>
  inline const dealii::SymmetricTensor<4, dim, Number> &
  ConstituentFiber<dim, Number>::get_tangent() const
  {
    return material->get_tangent();
  }

  template <int dim, typename Number>
  inline const dealii::Tensor<2, dim, Number> &
  ConstituentFiber<dim, Number>::get_prestretch_tensor() const
  {
    return prestretch_tensor;
  }

  template <int dim, typename Number>
  inline dealii::Tensor<2, dim, Number> &
  ConstituentFiber<dim, Number>::get_prestretch_tensor()
  {
    return prestretch_tensor;
  }

  template <int dim, typename Number>
  inline Number
  ConstituentFiber<dim, Number>::get_mass_fraction_ratio() const
  {
    return mass_fraction_ratio;
  }

  template <int dim, typename Number>
  inline TransferableParameters<Number> &
  ConstituentFiber<dim, Number>::get_transferable_parameters()
  {
    return transferable_parameters;
  }

  template <int dim, typename Number>
  inline const dealii::Tensor<1, dim, Number> &
  ConstituentFiber<dim, Number>::get_fiber_direction() const
  {
    return material->get_fiber_direction();
  }

  template <int dim, typename Number>
  [[nodiscard]] inline const std::string &
  ConstituentFiber<dim, Number>::get_name() const
  {
    return name;
  }

  template <int dim, typename Number>
  [[nodiscard]] inline unsigned int
  ConstituentFiber<dim, Number>::get_constituent_id() const
  {
    return constituent_id;
  }

  template <int dim, typename Number>
  inline void
  ConstituentFiber<dim, Number>::update_growth_and_remodeling(
    const dealii::Tensor<2, dim, Number> &F,
    const dealii::Tensor<2, dim, Number> &F_g_inv,
    const Number                          initial_mass_fraction)
  {
    // the factor J_g/J = 1/J_e tracks changes in the intrinsic mass density of
    // a constituent and it is assumed that all constituents have the same
    // relative changes in intrinsic mass density Famaey2023 (eq. 30). In the
    // case of an incompressible mixture, J_g/J approaches 1. The assumption of
    // a constant intrinsic mass density of a constituent is only valid for an
    // incompressible mixture see for example Latorre2020.
    const Number delta_sigma = material->get_fiber_cauchy_stress() - sig_h;

    {
      // update growth and remodeling
      update_growth(delta_sigma, initial_mass_fraction);

      // implicit update of remodeling
      const bool coupled = initial_mass_fraction > 0.;

      // update remodeling, use delta_sigma or delta_sigma_rot?
      update_remodeling(F, F_g_inv, delta_sigma, coupled);
    }
  }

  template <int dim, typename Number>
  void
  ConstituentFiber<dim, Number>::update_remodeling(
    const dealii::Tensor<2, dim, Number> &F,
    const dealii::Tensor<2, dim, Number> &F_g_inv,
    const Number                          delta_sigma,
    const bool                            coupled)
  {
    // update remodeling stretch
    current_lambda_r +=
      dt *
      evaluate_lambda_r_dot(F, F_g_inv, current_lambda_r, delta_sigma, coupled);

    // update inverse remodeling deformation gradient based on updated
    // current_lambda_r
    update_F_r_inv();
  }

  // Update growth. If initial_mass_fraction is < 0.0, switch to classical
  // model. This is automatically done if the mixture problem is setup with
  // coupled = false.
  template <int dim, typename Number>
  inline void
  ConstituentFiber<dim, Number>::update_growth(
    const Number delta_sigma,
    const Number initial_mass_fraction)
  {
    // explicit integration of mass fraction ratio
    if (initial_mass_fraction < 0.0)
      mass_fraction_ratio +=
        k_sig * mass_fraction_ratio * dt * (delta_sigma / sig_h);

    //(void) delta_sigma;
    // IMPORTANT to update the growth scalar in transferable parameters because
    // that is used in update_remodeling()! Alternatively, used
    // this->mass_fraction_ratio in update_remodeling()
    // transferable_parameters.current_mass_fraction *= 1.0 /
    // initial_mass_fraction; UPDATE(11/25/22): the comment above can be removed
    else
      mass_fraction_ratio =
        transferable_parameters.current_mass_fraction / initial_mass_fraction;
  }

  template <int dim, typename Number>
  inline void
  ConstituentFiber<dim, Number>::update_F_r_inv()
  {
    // update F_r_inv
    F_r_inv =
      lambda_pre / current_lambda_r * material->get_structural_tensor() +
      std::sqrt(current_lambda_r / lambda_pre) *
        material->get_orthogonal_structural_tensor();
    // update lambda_r in transferable parameters (for outputting)
    transferable_parameters.lambda_rem = current_lambda_r;
  }

  template <int dim, typename Number>
  inline void
  ConstituentFiber<dim, Number>::update_transferable_parameters()
  {
    transferable_parameters.sigma_hom  = sig_h;
    transferable_parameters.lambda_rem = current_lambda_r;
  }

  template <int dim, typename Number>
  inline void
  ConstituentFiber<dim, Number>::update_sigma_h()
  {
    // compute (elastic) right Cauchy Green based on prestretch tensor
    const auto C =
      dealii::Physics::Elasticity::Kinematics::C(prestretch_tensor);
    // set sigma_h based on prestretch
    sig_h = material->evaluate_fiber_cauchy_stress(
      C, transferable_parameters.contractility);
  }

  template <int dim, typename Number>
  inline dealii::Tensor<2, dim, Number>
  ConstituentFiber<dim, Number>::create_prestretch_tensor()
  {
    return lambda_pre * this->material->get_structural_tensor() +
           std::sqrt(1.0 / lambda_pre) *
             this->material->get_orthogonal_structural_tensor();
  }

  // instantiations
  template class ConstituentFiber<3, double>;

} // namespace Mixture::Constituents