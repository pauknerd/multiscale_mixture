#include <MSHCMM/mixture/constituents/constituent_hyperelastic.h>

namespace Mixture::Constituents
{

  template <int dim, typename Number>
  ConstituentHyperelastic<dim, Number>::ConstituentHyperelastic(
    const unsigned int                                    constituent_id,
    const dealii::Tensor<2, dim, Number>                 &prestretch_tensor,
    dealii::TensorFunction<2, dim, Number>               &prestretch_function,
    std::unique_ptr<Materials::MaterialBase<dim, Number>> material)
    : constituent_id(constituent_id)
    , prestretch_tensor(prestretch_tensor)
    , prestretch_function(prestretch_function)
    , material(std::move(material))
  {}

  template <int dim, typename Number>
  inline void
  ConstituentHyperelastic<dim, Number>::update_material(
    const dealii::Tensor<2, dim, Number> &F,
    const dealii::Tensor<2, dim, Number> &F_g_inv)
  {
    // total inelastic deformation gradient is F_g_inv * prestretch
    const auto F_in_inv = F_g_inv * prestretch_tensor;
    // update Material with F and F_in_inv
    material->update_material_data(F, F_in_inv);
  }

  template <int dim, typename Number>
  inline void
  ConstituentHyperelastic<dim, Number>::update_GR(
    const dealii::Tensor<2, dim, Number> &F,
    const dealii::Tensor<2, dim, Number> &F_g_inv,
    const Number                          initial_mass_fraction)
  {
    // do nothing
    (void)F;
    (void)F_g_inv;

    // only update if it was set by a transformer
    if (transferable_parameters.current_mass_fraction != 0.0)
      mass_fraction_ratio =
        transferable_parameters.current_mass_fraction / initial_mass_fraction;
  }

  template <int dim, typename Number>
  void
  ConstituentHyperelastic<dim, Number>::evaluate_prestretch(
    const dealii::Point<dim> &p)
  {
    // update prestretch tensor
    prestretch_tensor = prestretch_function.value(p) * prestretch_tensor;
  }

  template <int dim, typename Number>
  inline const dealii::SymmetricTensor<2, dim, Number> &
  ConstituentHyperelastic<dim, Number>::get_stress() const
  {
    return material->get_stress();
  }

  template <int dim, typename Number>
  inline const dealii::SymmetricTensor<4, dim, Number> &
  ConstituentHyperelastic<dim, Number>::get_tangent() const
  {
    return material->get_tangent();
  }

  template <int dim, typename Number>
  inline const dealii::Tensor<2, dim, Number> &
  ConstituentHyperelastic<dim, Number>::get_prestretch_tensor() const
  {
    return prestretch_tensor;
  }

  template <int dim, typename Number>
  inline dealii::Tensor<2, dim, Number> &
  ConstituentHyperelastic<dim, Number>::get_prestretch_tensor()
  {
    return prestretch_tensor;
  }

  template <int dim, typename Number>
  inline Number
  ConstituentHyperelastic<dim, Number>::get_mass_fraction_ratio() const
  {
    return mass_fraction_ratio;
  }

  template <int dim, typename Number>
  inline TransferableParameters<Number> &
  ConstituentHyperelastic<dim, Number>::get_transferable_parameters()
  {
    return transferable_parameters;
  }

  template <int dim, typename Number>
  inline const dealii::Tensor<1, dim, Number> &
  ConstituentHyperelastic<dim, Number>::get_fiber_direction() const
  {
    AssertThrow(false,
                dealii::ExcMessage(
                  "Hyperelastic Materials don't have a fiber direction!"));
  }

  template <int dim, typename Number>
  [[nodiscard]] inline const std::string &
  ConstituentHyperelastic<dim, Number>::get_name() const
  {
    return name;
  }

  template <int dim, typename Number>
  [[nodiscard]] inline unsigned int
  ConstituentHyperelastic<dim, Number>::get_constituent_id() const
  {
    return constituent_id;
  }

  // instantiations
  template class ConstituentHyperelastic<3, double>;

} // namespace Mixture::Constituents
