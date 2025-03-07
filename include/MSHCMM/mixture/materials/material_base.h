#pragma once

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

namespace Materials
{
  using namespace dealii;

  template <int dim, typename Number>
  class MaterialBase
  {
  public:
    virtual ~MaterialBase() = default;

    virtual void
    update_material_data(const Tensor<2, dim, Number> &F,
                         const Tensor<2, dim, Number> &F_inelastic) = 0;

    virtual const SymmetricTensor<2, dim, Number> &
    get_stress() const = 0;

    virtual const SymmetricTensor<4, dim, Number> &
    get_tangent() const = 0;
  };
} // namespace Materials
