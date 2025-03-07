#pragma once

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

namespace Materials
{

  using namespace dealii;

  template <int dim, typename Number>
  class FiberMaterialBase
  {
  public:
    virtual ~FiberMaterialBase() = default;

    virtual void
    update_material_data(const Tensor<2, dim, Number> &F,
                         const Tensor<2, dim, Number> &F_inelastic,
                         const Number                  contractility) = 0;

    virtual Number
    get_fiber_cauchy_stress() const = 0;

    virtual Number
    get_active_fiber_cauchy_stress() const = 0;

    virtual Number
    evaluate_fiber_cauchy_stress(
      const dealii::SymmetricTensor<2, dim, Number> &C_elastic,
      const Number                                   contractility) const = 0;

    virtual const SymmetricTensor<2, dim, Number> &
    get_structural_tensor() const = 0;

    virtual const SymmetricTensor<2, dim, Number> &
    get_orthogonal_structural_tensor() const = 0;

    virtual const Tensor<1, dim, Number> &
    get_fiber_direction() const = 0;

    virtual SymmetricTensor<2, dim, Number>
    get_dsig_dCe(const SymmetricTensor<2, dim, Number> &C_elastic) const = 0;

    virtual Number
    get_dsig_dlambdae(const Number I_4) const = 0;

    virtual const SymmetricTensor<2, dim, Number> &
    get_stress() const = 0;

    virtual const SymmetricTensor<4, dim, Number> &
    get_tangent() const = 0;
  };
} // namespace Materials
