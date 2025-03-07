#pragma once

#include <MSHCMM/mixture/prestretch_strategies/prestretch_strategy_base.h>

namespace Mixture::PrestretchStrategies
{

  template <int dim, typename Number = double>
  class PrestretchStrategyNone : public PrestretchStrategyBase<dim, Number>
  {
  public:
    PrestretchStrategyNone() = default;

    void
    update_prestretch(
      dealii::Tensor<2, dim, Number>       &prestretch_tensor,
      const dealii::Tensor<2, dim, Number> &F,
      const dealii::Point<dim>             &p = dealii::Point<dim>()) override
    {
      // do nothing
      (void)prestretch_tensor;
      (void)F;
      (void)p;
    }
  };
} // namespace Mixture::PrestretchStrategies
