#ifndef MSHCMM_DIFFUSION_MANAGER_H
#define MSHCMM_DIFFUSION_MANAGER_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/sundials/arkode.h>

#include <MSHCMM/common/boundary_descriptor.h>
#include <MSHCMM/common/fe_data.h>
#include <MSHCMM/mixture/growth_and_remodeling/local_mixture.h>
#include <MSHCMM/pathways/cells/local_cell_collection.h>
#include <MSHCMM/pathways/endothelial_cell_layer.h>

#include <fstream>
#include <functional>
#include <map>
#include <string>

#include "MSHCMM/mixture/constituents/cylindrical_coordinate_transformer.h"
#include "MSHCMM/pathways/pathway_manager.h"
#include "arkode/arkode_arkstep.h"
#include "diffusion_parameters.h"
#include "diffusion_parameters_constant.h"
#include "diffusion_parameters_pathway.h"
#include "iteration_tracker.h"

using namespace dealii;
using namespace Mixture;

namespace Diffusion
{
  /**
   * @brief Typedef of input transformer for the diffusion problem.
   *
   * Note: Removed the separate entries of the linear and quadratic matrix and
   * just pass `local_diffusion_parameters`. In the transformer you then call
   * ->get_K(), ->get_Q(), etc.
   *
   */
  template <int dim, typename Number = double>
  using InputTransformerDiffusion = std::function<
    void(const dealii::Point<dim>         &p,
         const Number                      time,
         const dealii::Vector<Number>     &diffusion_values,
         const std::vector<Number>        &average_pathway_output,
         const std::vector<Number>        &average_baseline_pathway_output,
         LocalDiffusionParameters<Number> &local_diffusion_parameters)>;

  template <int dim,
            typename VectorType,
            typename MatrixType,
            typename Number = double>
  class DiffusionManager
  {
  public:
    DiffusionManager(
      const dealii::parallel::TriangulationBase<dim> &triangulation,
      const unsigned int                              fe_degree,
      const unsigned int                              quad_degree,
      const unsigned int                              n_components,
      const std::shared_ptr<dealii::Function<dim>>   &initial_condition = {},
      const std::shared_ptr<dealii::Function<dim>>   &source_term       = {},
      const Common::BoundaryDescriptor<dim>          &boundary_descriptor =
        Common::BoundaryDescriptor<dim>());

    /**
     * @brief Setup vectors and matrices as well as LocalDiffusionParameters.
     *
     * @param diffusion_coefficients
     * @param linear_coefficients
     * @param higher_order_coefficients
     * @param higher_order_coefficients_derivative
     * @param enable_pathway_dependence if false, parameters are assumed constant.
     */
    void
    setup_system(
      const std::shared_ptr<dealii::Function<dim>> &diffusion_coefficients = {},
      const std::vector<std::vector<Number>>       &linear_coefficients    = {},
      const HigherOrderMatrix<Number> &higher_order_coefficients           = {},
      const HigherOrderMatrix<Number> &higher_order_coefficients_derivative =
        {},
      const bool enable_pathway_dependence = false);

    void
    assemble_mass_matrix();

    /**
     * @brief Assemble laplace matrix.
     *
     * @param d Function to describe the potentially spatially varying diffusion coefficient.
     */
    void
    assemble_laplace_matrix(const Function<dim, Number> *const d = nullptr);

    /**
     * @brief Assemble laplace matrix for anisotropic diffusion, potentially in a different coordinate system.
     */
    void
    assemble_laplace_matrix(
      const std::vector<std::shared_ptr<TensorFunction<2, dim, Number>>> &d,
      const Mixture::Constituents::CylindricalCoordinateTransformer<dim, Number>
        &cos_transformer);

    void
    assemble_source_and_boundary_terms(const Number time,
                                       const bool   use_single_cell_collection);

    void
    assemble_source_term(const FEValues<dim>           &fe_values,
                         const std::vector<Point<dim>> &quadrature_points,
                         Vector<Number>                &element_rhs);

    void
    assemble_neumann_bc(const FEFaceValues<dim>       &fe_face_values,
                        const Function<dim, Number>   &function,
                        const std::vector<Point<dim>> &quadrature_points,
                        Vector<Number>                &element_rhs) const;

    dealii::Vector<Number>
    assemble_endothelial_contribution(
      Pathways::PathwayManager<dim, Number> &pathway_manager);


    dealii::Vector<Number>
    assemble_endothelial_face(
      const dealii::FEFaceValues<dim> &fe_face_values,
      const std::vector<
        std::shared_ptr<Pathways::Cells::LocalCellCollection<dim, Number>>>
        &local_endothelial_cell_collection,
      const Pathways::OutputTransformerEndothelialPathway<dim, Number>
                                      &transformer,
      const Pathways::Cells::CellType &cell_type,
      dealii::Vector<Number>          &element_rhs) const;

    /**
     * Assemble outflux boundary condition based on the influxes resulting from
     * endothelial pathways
     */
    void
    assemble_outlfux_boundary(const dealii::Vector<Number> &influx_values,
                              const double                  outflux_fraction,
                              const unsigned int outflux_boundary_id);

    void
    assemble_outflux_face(const dealii::FEFaceValues<dim> &fe_face_values,
                          const dealii::Vector<Number>    &outflux,
                          dealii::Vector<Number>          &element_rhs) const;

    /**
     * @brief Assemble matrix with production and degradation.
     *
     * @param current_solution
     */
    void
    assemble_K(const VectorType &current_solution);

    /**
     * @brief Assemble jacobian matrix.
     *
     * @param current_solution
     */
    void
    assemble_jacobian(const VectorType &current_solution);

    void
    interpolate_initial_conditions();

    void
    output_results(const double time, const unsigned int time_step);

    void
    print_timer_stats(const bool print_mpi_stats) const
    {
      pcout << "DiffusionManager Timings";
      // always print summary
      computing_timer.print_summary();
      // optionally, print mpi stats
      if (print_mpi_stats)
        computing_timer.print_wall_time_statistics(mpi_communicator);
    }

    std::unique_ptr<DiffusionParameters<dim, Number>> &
    get_diffusion_parameters()
    {
      return diffusion_parameters;
    }

    void
    update_source_term_time(const Number time)
    {
      if (source_term)
        source_term->set_time(time);
    }

    const Common::FEData<dim, Number> &
    get_fe_data() const
    {
      return fe_data;
    }

    VectorType
    get_locally_relevant_solution() const
    {
      VectorType locally_relevant_solution;
      fe_data.initialize_locally_relevant_vector(locally_relevant_solution);
      locally_relevant_solution = solution;

      return locally_relevant_solution;
    }

    void
    add_transformer(const Pathways::Cells::CellType              &cell_type,
                    const InputTransformerDiffusion<dim, Number> &transformer)
    {
      // only insert transformer if not present yet since a cell type cannot
      // have multiple transformers
      auto [it, inserted] = transformers.insert({cell_type, transformer});
      // this asserts there is only one transformer per cell type
      Assert(inserted == true,
             dealii::ExcMessage(
               "There already is a transformer for the cell type '" +
               Pathways::Cells::CellType2string(cell_type) +
               "'! The "
               "DiffusionProblem only allows one transformer per cell type."));
    }

    // get quadrature point data of element
    std::vector<std::shared_ptr<LocalDiffusionParameters<Number>>>
    get_local_diffusion_parameters_cell(
      const typename dealii::Triangulation<dim>::cell_iterator &element)
    {
      return diffusion_parameters->get_local_diffusion_parameters(element);
    }

    void
    update_local_diffusion_parameters(
      const std::vector<dealii::Point<dim>>     &points,
      const Number                               time,
      const std::vector<dealii::Vector<Number>> &diffusion_values,
      const std::vector<
        std::shared_ptr<Pathways::Cells::LocalCellCollection<dim, Number>>>
        &local_cell_collection,
      std::vector<std::shared_ptr<LocalDiffusionParameters<Number>>>
        &local_diffusion_parameters);

    void
    set_output_filename(const std::string &filename,
                        const std::string &output_directory = "./")
    {
      output_filename = filename;
      output_dir      = output_directory;
    }

    void
    set_component_names(const std::vector<std::string> &component_names)
    {
      Assert(
        output_component_names.size() == component_names.size(),
        dealii::ExcMessage(
          "Given vector with component names does not have the right size!"));

      output_component_names = component_names;
    }

    MatrixType mass_matrix;
    MatrixType laplace_matrix;
    // degradation matrix
    MatrixType K;
    // derivative of K
    MatrixType jacobian;
    // complete system matrix: laplace + jacobian
    MatrixType system_matrix;
    // source terms and boundary values
    VectorType system_rhs;

    // locally owned solution
    VectorType solution;

    // preconditioner
    std::unique_ptr<TrilinosWrappers::PreconditionAMG> preconditioner;
    std::unique_ptr<TrilinosWrappers::PreconditionAMG> preconditioner_mass;

  private:
    bool is_setup{false};

    std::string              output_filename{"solution_diffusion"};
    std::string              output_dir{"./"};
    std::vector<std::string> output_component_names;

    MPI_Comm                   mpi_communicator;
    dealii::ConditionalOStream pcout;
    dealii::TimerOutput        computing_timer;

    // container with everything related to finite elements
    Common::FEData<dim, Number> fe_data;

    // The DiffusionParameters class (and its implementations) manage the
    // diffusion parameters for the different cases
    std::unique_ptr<DiffusionParameters<dim, Number>> diffusion_parameters;

    // transformers to get the right inputs from the pathways
    std::unordered_map<Pathways::Cells::CellType,
                       InputTransformerDiffusion<dim, Number>>
      transformers;

    // initial conditions
    std::shared_ptr<dealii::Function<dim>> initial_condition;

    // initial conditions
    std::shared_ptr<dealii::Function<dim>> source_term;

    // add boundary descriptor
    Common::BoundaryDescriptor<dim> boundary_descriptor;

    std::unique_ptr<SUNDIALS::ARKode<VectorType>> time_stepper;
  };
} // namespace Diffusion


#endif // MSHCMM_DIFFUSION_MANAGER_H
