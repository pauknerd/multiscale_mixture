#ifndef MSHCMM_DIFFUSION_PARAMETERS_CONSTANT_H
#define MSHCMM_DIFFUSION_PARAMETERS_CONSTANT_H

#include <deal.II/grid/tria.h>

#include "diffusion_parameters.h"

namespace Diffusion
{
  /**
   * Implementation of the DiffusionParameter interface for constant parameters.
   *
   * This class can be used if the DiffusionParameters are constant on each cell
   * and are not influenced by an external object. This class just creates a
   * "mock" CellDataStorage by creating a vector of shared_ptr of the size of
   * n_quad_points_diffusion. This vector is return vor each element, regardless
   * of what element is passed.
   *
   * @tparam dim spatial dimension of the problem
   * @tparam Number
   */
  template <int dim, typename Number = double>
  class DiffusionParametersConstant : public DiffusionParameters<dim, Number>
  {
  public:
    /**
     * Set diffusion coefficients, linear, and quadratic terms.
     *
     * @pre Assumes that all the values passed have the correct size which is
     * the number of species involved in the diffusion problem.
     *
     * @param K linear factors.
     * @param Q higher order terms.
     */
    DiffusionParametersConstant(const std::vector<std::vector<Number>> &K,
                                const HigherOrderMatrix<Number>        &Q,
                                const HigherOrderMatrix<Number> &Q_derivative)
      : constant_parameter_protoype(
          std::make_shared<LocalDiffusionParameters<Number>>(K,
                                                             Q,
                                                             Q_derivative))
    {}

    //! Store @param n_quad_points copies of the constant diffusion parameters.
    void
    set_size(const unsigned int n_quad_points) override
    {
      // store size
      n_quad_points_diffusion = n_quad_points;
      // assign
      local_diffusion_parameters.assign(n_quad_points,
                                        constant_parameter_protoype);
    }

    std::vector<std::shared_ptr<LocalDiffusionParameters<Number>>>
    get_local_diffusion_parameters(
      const typename dealii::Triangulation<dim>::cell_iterator &element)
      override
    {
      (void)element;
      return local_diffusion_parameters;
    }

  private:
    unsigned int n_quad_points_diffusion{0};
    std::vector<std::shared_ptr<LocalDiffusionParameters<Number>>>
      local_diffusion_parameters;

    // prototype of constant LocalDiffusionParameters
    std::shared_ptr<LocalDiffusionParameters<Number>>
      constant_parameter_protoype;
  };
} // namespace Diffusion

#endif // MSHCMM_DIFFUSION_PARAMETERS_CONSTANT_H
