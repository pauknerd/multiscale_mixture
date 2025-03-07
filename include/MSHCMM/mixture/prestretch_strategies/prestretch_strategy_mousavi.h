#pragma once

#include <MSHCMM/mixture/prestretch_strategies/prestretch_strategy_base.h>

namespace Mixture::PrestretchStrategies
{

  template <int dim, typename Number = double>
  class PrestretchStrategyMousavi : public PrestretchStrategyBase<dim, Number>
  {
  public:
    PrestretchStrategyMousavi() = default;

    void
    update_prestretch(
      dealii::Tensor<2, dim, Number>       &prestretch_tensor,
      const dealii::Tensor<2, dim, Number> &F,
      const dealii::Point<dim>             &p = dealii::Point<dim>()) override
    {
      (void)p;
      // update prestretch based on Mousavi et al., 2017
      prestretch_tensor = F * prestretch_tensor;
    }
  };
} // namespace Mixture::PrestretchStrategies
