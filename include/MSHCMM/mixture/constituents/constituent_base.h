#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <MSHCMM/mixture/materials/fiber_material_base.h>
#include <MSHCMM/mixture/materials/material_base.h>

#include "transferable_parameters.h"

namespace Mixture::Constituents
{

  template <int dim, typename Number = double>
  class ConstituentBase
  {
  public:
    virtual ~ConstituentBase() = default;

    virtual void
    update_material(const dealii::Tensor<2, dim, Number> &F,
                    const dealii::Tensor<2, dim, Number> &F_g_inv) = 0;

    // note that initial_mass_fraction is only needed in the case the HCMM is
    // coupled to a diffusion problem
    virtual void
    update_GR(const dealii::Tensor<2, dim, Number> &F,
              const dealii::Tensor<2, dim, Number> &F_g_inv,
              const Number                          initial_mass_fraction) = 0;

    virtual void
    evaluate_prestretch(const dealii::Point<dim> &p) = 0;

    virtual const dealii::SymmetricTensor<2, dim, Number> &
    get_stress() const = 0;

    virtual const dealii::SymmetricTensor<4, dim, Number> &
    get_tangent() const = 0;

    virtual const dealii::Tensor<2, dim, Number> &
    get_prestretch_tensor() const = 0;

    virtual dealii::Tensor<2, dim, Number> &
    get_prestretch_tensor() = 0;

    //! return the ratio of the current mass fraction to the initial mass
    //! fraction
    virtual Number
    get_mass_fraction_ratio() const = 0;

    virtual TransferableParameters<Number> &
    get_transferable_parameters() = 0;

    virtual const dealii::Tensor<1, dim, Number> &
    get_fiber_direction() const = 0;

    [[nodiscard]] virtual unsigned int
    get_constituent_id() const = 0;

    [[nodiscard]] virtual const std::string &
    get_name() const = 0;
  };
} // namespace Mixture::Constituents
