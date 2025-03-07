#pragma once

#include <deal.II/base/tensor.h>

namespace Mixture::PrestretchStrategies
{
  //! Typedef of prestretch function.
  template <int dim, typename Number = double>
  using PrestretchFunction =
    std::function<void(dealii::Tensor<2, dim, Number>       &prestretch_tensor,
                       const dealii::Tensor<2, dim, Number> &F)>;

  template <int dim, typename Number = double>
  class PrestretchStrategyBase
  {
  public:
    virtual ~PrestretchStrategyBase() = default;

    virtual void
    update_prestretch(
      dealii::Tensor<2, dim, Number>       &prestretch_tensor,
      const dealii::Tensor<2, dim, Number> &F,
      const dealii::Point<dim, Number> &p = dealii::Point<dim, Number>()) = 0;
  };
} // namespace Mixture::PrestretchStrategies
