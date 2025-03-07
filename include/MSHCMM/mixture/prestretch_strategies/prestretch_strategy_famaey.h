#pragma once

#include <MSHCMM/mixture/prestretch_strategies/prestretch_strategy_base.h>

#include "MSHCMM/mixture/constituents/cylindrical_coordinate_transformer.h"

namespace Mixture::PrestretchStrategies
{

  template <int dim, typename Number = double>
  class PrestretchStrategyFamaey : public PrestretchStrategyBase<dim, Number>
  {
  public:
    explicit PrestretchStrategyFamaey(
      const Mixture::Constituents::CylindricalCoordinateTransformer<dim>
                &cos_transformer,
      const bool incompressible = false)
      : cos_transformer(cos_transformer)
      , incompressible(incompressible)
    {}

    void
    update_prestretch(dealii::Tensor<2, dim, Number>       &prestretch_tensor,
                      const dealii::Tensor<2, dim, Number> &F,
                      const dealii::Point<dim, Number>     &p =
                        dealii::Point<dim, Number>()) override
    {
      auto F_mix_mod = F;
      if (incompressible)
        { // rotate F_mixture from global cartesian COS to local cylindrical COS
          auto F_mix_rot = cos_transformer.from_cartesian(p, F);
          // modify entries of F_mix_rot to remove shear terms
          F_mix_rot[0][1] = 0.0;
          F_mix_rot[0][2] = 0.0;
          F_mix_rot[1][0] = 0.0;
          F_mix_rot[1][2] = 0.0;
          F_mix_rot[2][0] = 0.0;
          F_mix_rot[2][1] = 0.0;
          // set radial prestretch such that F_mix_rot has determinant of 1
          // (isochoric) based on circumferential and axial stretches
          F_mix_rot[0][0] = 1.0 / (F_mix_rot[1][1] * F_mix_rot[2][2]);
          // rotate F_mix_rot back to global cartesian coordinate system
          F_mix_mod = cos_transformer.to_cartesian(p, F_mix_rot);
        }

      // multiply existing prestretch with modified deformation gradient
      prestretch_tensor = F_mix_mod * prestretch_tensor;
    }

  private:
    const Mixture::Constituents::CylindricalCoordinateTransformer<dim>
        &cos_transformer;
    bool incompressible;
  };
} // namespace Mixture::PrestretchStrategies
