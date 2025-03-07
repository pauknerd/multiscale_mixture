#pragma once

#include "constituent_base.h"

namespace Mixture::Constituents
{

  //! implementation of concrete constituent
  template <int dim, typename Number = double>
  class ConstituentHyperelastic : public ConstituentBase<dim, Number>
  {
  public:
    explicit ConstituentHyperelastic(
      const unsigned int                                    constituent_id,
      const dealii::Tensor<2, dim, Number>                 &prestretch_tensor,
      dealii::TensorFunction<2, dim, Number>               &prestretch_function,
      std::unique_ptr<Materials::MaterialBase<dim, Number>> material);

    void
    update_material(const dealii::Tensor<2, dim, Number> &F,
                    const dealii::Tensor<2, dim, Number> &F_g_inv) override;

    void
    update_GR(const dealii::Tensor<2, dim, Number> &F,
              const dealii::Tensor<2, dim, Number> &F_g_inv,
              const Number initial_mass_fraction) override;

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

    // not really used in hyperelastic constituent (eg. elastin)? Might be
    // useful if elastin is degraded though...
    TransferableParameters<Number> &
    get_transferable_parameters() override;

    // NOTE: return zero tensor as this does not apply
    const dealii::Tensor<1, dim, Number> &
    get_fiber_direction() const override;

    [[nodiscard]] const std::string &
    get_name() const override;

    [[nodiscard]] unsigned int
    get_constituent_id() const override;

  private:
    // name
    std::string name{"hyperelastic"};
    // constituent id
    unsigned int constituent_id;
    // transferable parameters todo: not sure if that is needed in the
    // hyperelastic constituent since this usually doesn't grow or remodel?
    TransferableParameters<Number> transferable_parameters;
    // mass fraction ratio, ratio of current mass fraction to initial mass
    // fraction todo: should this be part of the transferable_parameters?
    Number mass_fraction_ratio{1.0};
    // prestretch tensor
    dealii::Tensor<2, dim, Number> prestretch_tensor;
    // prestretch function
    dealii::TensorFunction<2, dim, Number> &prestretch_function;
    // material
    std::unique_ptr<Materials::MaterialBase<dim, Number>> material;
  };
} // namespace Mixture::Constituents
