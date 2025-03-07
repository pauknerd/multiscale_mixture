#ifndef MAIN_DIFFUSION_PARAMETERS_PATHWAY_H
#define MAIN_DIFFUSION_PARAMETERS_PATHWAY_H

#include <deal.II/base/quadrature_point_data.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria.h>

#include "diffusion_parameters.h"

namespace Diffusion
{
  /**
   * Implementation of the DiffusionParameter interface for parameters that can
   * be modified by PathwayEquations.
   *
   * This class can be used if the DiffusionParameters are influenced by the
   * solution of PathwayEquations. This class creates a CellDataStorage of the
   * size of n_quad_points_diffusion on each cell.
   *
   * @tparam dim spatial dimension of the problem
   * @tparam Number
   */
  template <int dim, typename Number = double>
  class DiffusionParametersPathway : public DiffusionParameters<dim, Number>
  {
  public:
    DiffusionParametersPathway(const dealii::Triangulation<dim> &triangulation,
                               const unsigned int n_quad_points_diffusion,
                               const unsigned int n_species_diffusion)
    {
      // initialize CellDataStorage
      distribute_local_diffusion_parameters(triangulation,
                                            n_quad_points_diffusion,
                                            n_species_diffusion);
    }

    DiffusionParametersPathway(const dealii::Triangulation<dim> &triangulation,
                               const unsigned int n_quad_points_diffusion,
                               const unsigned int n_species_diffusion,
                               const std::vector<std::vector<Number>> &K,
                               const HigherOrderMatrix<Number>        &Q,
                               const HigherOrderMatrix<Number> &Q_derivative)
    {
      // create prototype
      constant_parameter_protoype =
        std::make_shared<LocalDiffusionParameters<Number>>(K, Q, Q_derivative);
      // initialize CellDataStorage
      distribute_local_diffusion_parameters(triangulation,
                                            n_quad_points_diffusion,
                                            n_species_diffusion);
    }

    //! Not needed when diffusion parameters depend on pathways.
    void
    set_size(const unsigned int n_quad_points) override
    {
      (void)n_quad_points;
      // nothing needs to be done in this case.
    }

    //! return the local diffusion parameters on the @p element.
    std::vector<std::shared_ptr<LocalDiffusionParameters<Number>>>
    get_local_diffusion_parameters(
      const typename dealii::Triangulation<dim>::cell_iterator &element)
      override
    {
      return local_diffusion_parameters.get_data(element);
    }

  private:
    /**
     * @brief Initialize the CellDataStorage to the correct size and set the initial values.
     *
     * @param triangulation grid used in the simulation of the diffusion problem
     * @param n_quad_points_diffusion number of quadrature points in the diffusion problem
     * @param n_species_diffusion number of species used in the diffusion problem
     */
    void
    distribute_local_diffusion_parameters(
      const dealii::Triangulation<dim> &triangulation,
      const unsigned int                n_quad_points_diffusion,
      const unsigned int                n_species_diffusion)
    {
      std::cout << "Setting up local_diffusion_parameters data PATHWAY..."
                << std::endl;

      // create filter to extract locally owned cells
      using CellFilter =
        dealii::FilteredIterator<typename dealii::parallel::TriangulationBase<
          dim>::active_cell_iterator>;

      local_diffusion_parameters.initialize(
        CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
                   triangulation.begin_active()),
        CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
                   triangulation.end()),
        n_quad_points_diffusion);

      // loop over all locally owned elements
      for (const auto &element : triangulation.active_cell_iterators())
        {
          if (element->is_locally_owned())
            {
              // get so far empty local data on the element
              const std::vector<
                std::shared_ptr<LocalDiffusionParameters<Number>>>
                local_parameters = local_diffusion_parameters.get_data(element);

              // loop over number of quadrature points and resize the diffusion
              // parameters to have the proper size
              for (unsigned int q_point = 0; q_point < n_quad_points_diffusion;
                   ++q_point)
                {
                  if (constant_parameter_protoype)
                    {
                      local_parameters[q_point]->get_K() =
                        constant_parameter_protoype->get_K();
                      local_parameters[q_point]->get_Q() =
                        constant_parameter_protoype->get_Q();
                      local_parameters[q_point]->get_Q_derivative() =
                        constant_parameter_protoype->get_Q_derivative();
                    }
                  else
                    {
                      // resize
                      local_parameters[q_point]->resize_parameters(
                        n_species_diffusion);
                      // todo: set initial parameters IMPORTANT
                      local_parameters[q_point]->set_parameters();
                    }
                }
            }
        }
      std::cout << "Setting up local_diffusion_parameters data PATHWAY...DONE!"
                << std::endl;
    }

    // LocalDiffusionParameters at quadrature points
    dealii::CellDataStorage<typename dealii::Triangulation<dim>::cell_iterator,
                            LocalDiffusionParameters<Number>>
      local_diffusion_parameters;

    // prototype
    std::shared_ptr<LocalDiffusionParameters<Number>>
      constant_parameter_protoype;
  };
} // namespace Diffusion

#endif // MAIN_DIFFUSION_PARAMETERS_PATHWAY_H
