#ifndef MSHCMM_DIFFUSION_PARAMETERS_H
#define MSHCMM_DIFFUSION_PARAMETERS_H

#include "local_diffusion_parameters.h"

namespace Diffusion
{
  /**
   * Interface for LocalDiffusionParameters. The only function this interface
   * prescribes is what vector of LocalDiffusionParameters should be returned.
   * The basic assumption of the class is that the dealii::CellDataStorage class
   * is used to handle quadrature point data that potentially differs from point
   * to point.
   */
  template <int dim, typename Number = double>
  class DiffusionParameters
  {
  public:
    virtual ~DiffusionParameters() = default;

    /**
     * @brief Only needed in case of constant LocalDiffusionParameters to set the proper size by the
     *  diffusion problem.
     *
     * @param n_quad_points
     */
    virtual void
    set_size(const unsigned int n_quad_points) = 0;

    virtual std::vector<std::shared_ptr<LocalDiffusionParameters<Number>>>
    get_local_diffusion_parameters(
      const typename dealii::Triangulation<dim>::cell_iterator &element) = 0;

  private:
  };
} // namespace Diffusion

#endif // MSHCMM_DIFFUSION_PARAMETERS_H
