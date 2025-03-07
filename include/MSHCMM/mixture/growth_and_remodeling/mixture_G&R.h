#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/sundials/kinsol.h>

#include <deal.II/trilinos/nox.h>

#include <MSHCMM/common/boundary_descriptor.h>
#include <MSHCMM/mixture/constituents/constituent_factory_base.h>
#include <MSHCMM/mixture/growth_strategies/growth_strategy_base.h>
#include <MSHCMM/mixture/prestretch_strategies/prestretch_strategy_base.h>
#include <MSHCMM/mixture/solver/LinearSolver.h>
#include <MSHCMM/mixture/solver/SystemState.h>
#include <MSHCMM/mixture/solver/errors.h>
#include <MSHCMM/pathways/cells/local_cell_collection.h>
#include <MSHCMM/utilities/helpers.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>

#include "local_mixture.h"
#include "postprocessing_functions.h"
#include "postprocessor.h"

namespace Mixture
{

  template <int dim>
  using CellIteratorType = typename dealii::Triangulation<dim>::cell_iterator;


  //! Typedef of input transformer for the HCMM problem.
  template <int dim, typename Number = double>
  using InputTransformerHCMM = std::function<void(
    const dealii::Point<dim>     &p,
    const Number                  time,
    const dealii::Vector<Number> &diffusion_values,
    const dealii::Vector<Number> &diffusion_laplacians,
    const std::vector<Number>    &average_pathway_output,
    const std::vector<Number>    &average_baseline_pathway_output,
    Constituents::TransferableParameters<Number> &transferable_parameters)>;

  template <int dim,
            typename VectorType,
            typename MatrixType,
            typename Number = double>
  class Mixture_GR
  {
  public:
    explicit Mixture_GR(
      Common::FEData<dim, Number> &&fe_data,
      std::unique_ptr<GrowthStrategies::GrowthStrategyBase<dim, Number>>
        growth_strategy,
      std::unique_ptr<PrestretchStrategies::PrestretchStrategyBase<dim, Number>>
        prestretch_strategy,
      std::vector<
        std::unique_ptr<Constituents::ConstituentFactoryBase<dim, Number>>>
                                                     constituent_factories,
      const Common::BoundaryDescriptor<dim, Number> &new_boundary_descriptor,
      const bool                                     coupled = false);

    Mixture_GR(
      const dealii::Triangulation<dim> &triangulation,
      const unsigned int                fe_degree,
      const unsigned int                quad_degree,
      std::unique_ptr<GrowthStrategies::GrowthStrategyBase<dim, Number>>
        growth_strategy,
      std::unique_ptr<PrestretchStrategies::PrestretchStrategyBase<dim, Number>>
        prestretch_strategy,
      std::vector<
        std::unique_ptr<Constituents::ConstituentFactoryBase<dim, Number>>>
                                                     constituent_factories,
      const Common::BoundaryDescriptor<dim, Number> &new_boundary_descriptor,
      const bool                                     coupled = false);

    Mixture_GR(Mixture_GR<dim, VectorType, MatrixType, Number> &&) noexcept =
      default;

    void
    setup_system();

    /**
     * @brief Setup the quadrature point data for the local mixtures.
     *
     * @details
     *
     *
     * @param[in] initial_reference_density Function describing the initial
     * reference mass density of the mixture, possibly space dependent.
     * @param[in] initial_mass_fractions Map which contains a dealii::Function
     * describing the initial mass fraction ratios for each material id. Each
     * component of the functions describes the initial mass fraction of a
     * constituent. Note that the number of components must match the number of
     * constituents for a given material_id.
     */
    void
    setup_qp_data(
      const std::shared_ptr<dealii::Function<dim>> initial_reference_density,
      const std::map<unsigned int, std::shared_ptr<dealii::Function<dim>>>
        &initial_mass_fractions);

    /**
     * @brief Same as above but assumes the `initial_mass_fractions` are the same on all elements.
     *
     * @pre Assumes all constituents are distributed on elements with
     * material_id 0!
     *
     * @param initial_reference_density
     * @param initial_mass_fractions
     */
    void
    setup_qp_data(
      const std::shared_ptr<dealii::Function<dim>> initial_reference_density,
      const std::shared_ptr<dealii::Function<dim>> initial_mass_fractions);

    /**
     * @brief Assemble global tangent matrix and residual.
     */

    void
    assemble_residual(const VectorType &total_solution, VectorType &residual);

    void
    assemble_matrix(const VectorType &total_solution);

    //! update material data on all elements
    void
    update_material_data();

    void
    update_material_data(const VectorType &total_solution);

    //! @brief update only specified constituents
    void
    update_material_data(const std::vector<unsigned int> &constituent_ids);

    void
    update_GR();

    //! @brief Update prestretch of constituents that are prestressed iteratively
    void
    update_prestretch(const std::vector<unsigned int> &constituent_ids);

    //! @brief Update prestretch according to given functions in constituent factories.
    void
    evaluate_and_set_prestretch();

    /**
     * @brief Update transferable parameters of the constituents on an element based on diffusion solution and pathway outputs.
     *
     * @details
     *
     *
     * @pre Assumes mixture problem uses same quadrature rule as diffusion
     * problem.
     *
     * @param points Coordinates of points
     * @param time Current simulation time.
     * @param diffusion_values Current solution of the diffusion problem.
     * @param local_cell_collection Contains all the cell types and their current outputs.
     * @param local_mixture Contains all the constituents that can be updated.
     */
    void
    update_transferable_parameters(
      const std::vector<dealii::Point<dim>>     &points,
      const Number                               time,
      const std::vector<dealii::Vector<Number>> &diffusion_values,
      const std::vector<dealii::Vector<Number>> &diffusion_laplacians,
      const std::vector<
        std::shared_ptr<Pathways::Cells::LocalCellCollection<dim, Number>>>
        &local_cell_collection,
      std::vector<std::shared_ptr<LocalMixture<dim, Number>>> &local_mixture);

    // Debug only
    //! Print constituent names at quadrature points.
    void
    print_qp_data();

    void
    write_prestretched_configuration(const std::string &output_filename,
                                     const unsigned int constituent_id) const;

    /**
     * @brief Load an already prestretched configuration.
     *
     * @details
     * Usage: set up everything as usual, including the desired refinement level
     * of the grid (important for setting the proper sizes of vectors and
     * quadrature point storage). Set prestretch of fiber based constituents to
     * their final value (no need to ramp up, that was done during the
     * Prestretching), setup mixture with a call to `setup_system()` `and
     * setup_qp_data()`. Before starting the time loop of the simulation though,
     * call this function which will set the prestretch of the elastin to the
     * value found during Prestretching and the solution to the final solution.
     *
     * Includes the final solution vector and the prestretch tensors at the
     * quadrature points of the iteratively prestretched hyperelastic
     * constituent (usually elastin).
     *
     * Note that currently only one iteratively constituent is supported.
     *
     * Calls `update_material_data()` with the read in solution to set the
     * correct values in the constituents, such as stresses.
     *
     *
     * @param constituent_id Id of the constituent which was prestressed iteratively.
     */
    void
    read_prestretched_configuration(const unsigned int constituent_id);

    /**
     * @brief Get Postprocessor of Mixture.
     *
     * @return Reference to Postprocessor.
     */
    Postprocessing::Postprocessor<dim, Number> &
    get_postprocessor();

    /**
     * @brief Get BoundaryDescriptor of Mixture.
     *
     * @return Reference to BoundaryDescriptor.
     */
    Common::BoundaryDescriptor<dim, Number> &
    get_new_boundary_descriptor();

    void
    output_results(const unsigned int step);

    /**
     * @brief Add a transformer for a certain constituent.
     *
     *
     * @param constituent_id
     * @param cell_type
     * @param input_transformer
     */
    void
    add_transformer(const unsigned int                       constituent_id,
                    const Pathways::Cells::CellType         &cell_type,
                    const InputTransformerHCMM<dim, Number> &input_transformer);

    const Common::FEData<dim, Number> &
    get_fe_data() const;

    Common::FEData<dim, Number> &
    get_fe_data();

    std::vector<std::shared_ptr<LocalMixture<dim, Number>>>
    get_local_mixture(const CellIteratorType<dim> &element);

    const dealii::CellDataStorage<CellIteratorType<dim>,
                                  LocalMixture<dim, Number>> &
    get_local_mixtures() const;

    [[nodiscard]] bool
    is_coupled() const;

    const VectorType &
    get_solution() const;

    VectorType
    get_total_relevant_solution() const;

    void
    print_timer_stats(const bool print_mpi_stats = false) const;

    /**
     * @brief Solve a nonlinear step with the TRILINOS NOX solver.
     *
     * @param step
     * @param time
     * @param active_GR
     */
    void
    solve_step_with_NOX(const unsigned int step,
                        const Number       time,
                        const bool         active_GR)
    {
      Assert(
        nox_solver != nullptr,
        dealii::ExcMessage(
          "You are trying to use the NOX solver but it has not been setup! Make sure to call setup_NOX first!"));

      pcout << "Solving MIXTURE at time : " << time << " (step " << step << ")"
            << std::endl;

      // apply boundary conditions to current TOTAL solution
      boundary_descriptor.set_evaluation_time(time);
      boundary_descriptor.build_dirichlet_constraints(fe_data.get_dof_handler(),
                                                      fe_data.get_constraints(),
                                                      false);
      fe_data.get_constraints().distribute(system_state.solution);

      //      // DEBUG
      //      std::cout << "current solution:" << std::endl;
      //      system_state.solution.print(std::cout);
      //      std::cout << "predictor:" << std::endl;
      //      system_state.predictor.print(std::cout);

      // reset errors
      error_residual.reset();
      error_residual_0.reset();
      error_residual_norm.reset();

      // print header
      print_conv_header();

      // do actual solve
      nox_solver->solve(system_state.solution);

      // print final convergence footer
      print_conv_footer();

      // update G&R
      if (active_GR)
        update_GR();
    }

    /**
     * @brief Do a prestretch step with the NOX solver.
     *
     * @param step
     * @param constituent_ids
     * @param threshold
     * @return
     */
    bool
    do_prestretch_step_with_NOX(
      const unsigned int               step,
      const unsigned int               n_load_steps,
      const std::vector<unsigned int> &constituent_ids,
      const Number                     threshold,
      const bool                       check_max_nodal_displacement = false)
    {
      if (step <= n_load_steps)
        { // what to do before starting to solve
          // set time of prestretch functions in the constituent factories
          set_time_of_prestretch_functions(step);
          // evaluate and set prestretches (of all constituents based on
          // prescribed prestretch functions)
          evaluate_and_set_prestretch();
        }

      // solve step
      solve_step_with_NOX(step, step, false);

      // update prestretch of constituents that are prestretched iteratively
      // (i.e., with ids in constituent_ids).
      update_prestretch(constituent_ids);

      // check for convergence of Prestretching
      bool converged = false;
      if (check_max_nodal_displacement)
        {
          // check max nodal displacements
          const auto max_nodal_disp =
            HELPERS::get_max_nodal_displacement(fe_data.get_dof_handler(),
                                                get_total_relevant_solution());
          pcout << "Max global nodal displacement: " << max_nodal_disp
                << std::endl;

          converged = max_nodal_disp <= threshold;
        }
      else
        {
          const auto avg_nodal_disp =
            HELPERS::get_avg_nodal_displacement(fe_data.get_dof_handler(),
                                                get_total_relevant_solution());
          pcout << "Avg global nodal displacement: " << avg_nodal_disp
                << std::endl;

          converged = avg_nodal_disp <= threshold;
        }

      return converged;
    }

    /**
     * @brief Setup NOX solver with a direct solver.
     *
     * @param additional_data
     * @param linear_solver
     * @param mpi_comm
     */
    void
    setup_NOX(
      typename dealii::TrilinosWrappers::NOXSolver<VectorType>::AdditionalData
                                                 &nox_additional_data,
      const Teuchos::RCP<Teuchos::ParameterList> &parameters =
        Teuchos::rcp(new Teuchos::ParameterList),
      const MPI_Comm &mpi_comm = MPI_COMM_WORLD)
    {
      (void)mpi_comm;

      // create solver
      nox_solver =
        std::make_unique<dealii::TrilinosWrappers::NOXSolver<VectorType>>(
          nox_additional_data, parameters);

      // set the required functions in NOX solver
      nox_solver->residual = [&](const VectorType &evaluation_point,
                                 VectorType       &residual) {
        assemble_residual(evaluation_point, residual);

        return 0;
      };

      nox_solver->setup_jacobian = [&](const VectorType &current_u) {
        assemble_matrix(current_u);

        return 0;
      };

      // check iteration status, ONLY used to print residuals
      // that works since NOX uses OR combination to determine convergence
      nox_solver->check_iteration_status =
        [&](const unsigned int i,
            const double       norm_f,
            const VectorType  &current_u,
            const VectorType  &f) -> dealii::SolverControl::State {
        (void)norm_f;
        (void)current_u;

        // print newton statistics
        print_newton_stats(i, f);

        return dealii::SolverControl::State::iterate;
      };

      nox_solver->solve_with_jacobian =
        [&](const VectorType &rhs, VectorType &dst, const double tolerance) {
          (void)tolerance;

          {
            dealii::TimerOutput::Scope t(computing_timer,
                                         "Mixture - Solve linear system");

            dealii::SolverControl solver_control =
              dealii::SolverControl(1000, 1.e-12);

            typename dealii::TrilinosWrappers::SolverDirect::AdditionalData
              additional_data;

            dealii::TrilinosWrappers::SolverDirect solver(solver_control,
                                                          additional_data);
            solver.solve(system_state.system_matrix, dst, rhs);

            linear_solver_stats = std::make_pair(solver_control.last_step(),
                                                 solver_control.last_value());
          }
          // NOTE: important to use zero constraints here!
          fe_data.get_zero_constraints().distribute(dst);

          return 0;
        };
    }

    /**
     * @brief Setup NOX solver with an iterative solver.
     *
     * @param additional_data
     * @param linear_solver
     * @param mpi_comm
     */
    template <typename Preconditioner>
    void
    setup_NOX(
      IterativeSolver<VectorType, MatrixType> &linear_solver,
      Preconditioner                          &preconditioner,
      const typename Preconditioner::AdditionalData
        &preconditioner_additional_data,
      typename dealii::TrilinosWrappers::NOXSolver<VectorType>::AdditionalData
                                                 &nox_additional_data,
      const Teuchos::RCP<Teuchos::ParameterList> &parameters =
        Teuchos::rcp(new Teuchos::ParameterList),
      const MPI_Comm &mpi_comm = MPI_COMM_WORLD)
    {
      (void)mpi_comm;
      // create solver
      nox_solver =
        std::make_unique<dealii::TrilinosWrappers::NOXSolver<VectorType>>(
          nox_additional_data, parameters);

      // set the required functions in NOX solver
      nox_solver->residual = [&](const VectorType &evaluation_point,
                                 VectorType       &residual) {
        assemble_residual(evaluation_point, residual);

        return 0;
      };

      nox_solver->setup_jacobian = [&](const VectorType &current_u) {
        assemble_matrix(current_u);

        return 0;
      };

      // check iteration status
      nox_solver->check_iteration_status =
        [&](const unsigned int i,
            const double       norm_f,
            const VectorType  &current_u,
            const VectorType  &f) -> dealii::SolverControl::State {
        (void)norm_f;
        (void)current_u;

        print_newton_stats(i, f);

        // default return value
        return dealii::SolverControl::State::iterate;
      };

      nox_solver->solve_with_jacobian =
        [&](const VectorType &rhs, VectorType &dst, const double tolerance) {
          // setup preconditioner
          {
            dealii::TimerOutput::Scope t(computing_timer,
                                         "Mixture - Build preconditioner");

            // setup preconditioner
            preconditioner.initialize(system_state.system_matrix,
                                      preconditioner_additional_data);
          }
          // solve linear system
          {
            dealii::TimerOutput::Scope t(computing_timer,
                                         "Mixture - Solve linear system");
            // set tolerance based on what NOX determines
            linear_solver.get_solver_control().set_tolerance(tolerance);
            // solve
            linear_solver_stats = linear_solver.solve(
              system_state.system_matrix, dst, rhs, preconditioner);
          }
          // NOTE: important to use zero constraints here!
          fe_data.get_zero_constraints().distribute(dst);

          return 0;
        };
    }

  private:
    MPI_Comm                   mpi_communicator;
    dealii::ConditionalOStream pcout;
    dealii::TimerOutput        computing_timer;
    // reference volume of the entire domain
    Number reference_volume;
    // current volume
    Number                          current_vol;
    std::pair<Number, Number>       error_dil;
    std::pair<unsigned int, Number> linear_solver_stats;

    // container with everything related to finite elements
    Common::FEData<dim, Number> fe_data;
    // Growth strategy
    std::unique_ptr<GrowthStrategies::GrowthStrategyBase<dim, Number>>
      growth_strategy;
    // PrestretchingStrategy
    std::unique_ptr<PrestretchStrategies::PrestretchStrategyBase<dim, Number>>
      prestretch_strategy;
    // MixturePostprocessor
    Postprocessing::Postprocessor<dim, Number> mixture_postprocessor;

    // constituents present in the mixture
    std::vector<
      std::unique_ptr<Constituents::ConstituentFactoryBase<dim, Number>>>
      constituent_factories;
    // quadrature point data of local mixtures, i.e. each quadrature point has a
    // vector of constituents
    dealii::CellDataStorage<CellIteratorType<dim>, LocalMixture<dim, Number>>
      local_mixtures;
    // transformers from DiffusionManager and Pathways to HCMM
    std::unordered_map<
      unsigned int,
      std::vector<std::pair<Pathways::Cells::CellType,
                            InputTransformerHCMM<dim, Number>>>>
      transformers;

    // system matrix, rhs, solution, etc.
    SystemState<VectorType, MatrixType> system_state;
    // NOX Solver
    std::unique_ptr<dealii::TrilinosWrappers::NOXSolver<VectorType>>
      nox_solver{};

    // add boundary descriptor
    Common::BoundaryDescriptor<dim, Number> boundary_descriptor;
    // coupled to diffusion problem?
    const bool coupled;

    // Value extractors
    const dealii::FEValuesExtractors::Vector u_fe;

    Errors error_residual, error_residual_0, error_residual_norm;

    void
    get_error_residual(Errors &error_res, const VectorType &current_residual)
    {
      VectorType error_vec(fe_data.get_locally_owned_dofs(), mpi_communicator);

      for (const auto i : fe_data.get_dof_handler().locally_owned_dofs())
        if (!fe_data.get_constraints().is_constrained(i))
          error_vec(i) = current_residual(i);

      error_res.norm = error_vec.l2_norm();
      error_res.u    = error_vec.l2_norm();
    }

    void
    print_conv_header()
    {
      static const unsigned int l_width = 70;

      for (unsigned int i = 0; i < l_width; ++i)
        pcout << '_';
      pcout << std::endl;

      pcout << " STEP "
            << "|  LIN_IT  LIN_RES    R_ABS      R_U_ABS    R_NORM  "
            << "   R_U_NORM" << std::endl;

      for (unsigned int i = 0; i < l_width; ++i)
        pcout << '_';
      pcout << std::endl;
    }

    void
    print_conv_footer()
    {
      error_dil =
        std::make_pair(-1.0,
                       HELPERS::compute_current_volume(fe_data,
                                                       local_mixtures) /
                         reference_volume);

      pcout << std::endl
            << "v / V_0:\t" << error_dil.second * reference_volume << " / "
            << reference_volume << " = " << error_dil.second << std::endl;

      static const unsigned int l_width = 70;

      for (unsigned int i = 0; i < l_width; ++i)
        pcout << '_';
      pcout << std::endl << std::endl;
    }

    // print nonlinear solver statistics
    void
    print_newton_stats(const unsigned newton_iteration, const VectorType &f)
    {
      // compute residual
      get_error_residual(error_residual, f);

      if (newton_iteration == 0)
        error_residual_0 = error_residual;

      error_residual_norm = error_residual;
      error_residual_norm.normalize(error_residual_0);

      // print residuals
      pcout << std::setw(5) << newton_iteration << " | " << std::fixed
            << std::setprecision(3) << std::setw(7) << std::scientific
            << linear_solver_stats.first << "  " << linear_solver_stats.second
            << "  " << error_residual.norm << "  " << error_residual.u << "  "
            << error_residual_norm.norm << "  " << error_residual_norm.u << "  "
            << std::endl;
    }

    /**
     * @brief Update time in prestretch functions of constituents factories.
     */
    void
    set_time_of_prestretch_functions(const Number time) const;
  };

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  inline void
  Mixture_GR<dim, VectorType, MatrixType, Number>::add_transformer(
    const unsigned int                       constituent_id,
    const Pathways::Cells::CellType         &cell_type,
    const InputTransformerHCMM<dim, Number> &input_transformer)
  {
    // check if key (constituent_id) already exists, if yes, add pair of
    // cell_type and transformer to vector
    if (transformers.find(constituent_id) != transformers.end())
      transformers.at(constituent_id).push_back({cell_type, input_transformer});
    else
      transformers[constituent_id] = {
        std::make_pair(cell_type, input_transformer)};

    //    // DEBUG - print map
    //    for (const auto& [id, vector_of_transformers] : transformers)
    //      {
    //        std::cout << "transformers for constituent with " << id << ": ";
    //        for (const auto& [cell_t, transformer] : vector_of_transformers)
    //          {
    //            std::cout << Pathways::Cells::CellType2string(cell_t) << " ";
    //          }
    //        std::cout << std::endl;
    //      }
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  inline const VectorType &
  Mixture_GR<dim, VectorType, MatrixType, Number>::get_solution() const
  {
    return system_state.solution;
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  inline const Common::FEData<dim, Number> &
  Mixture_GR<dim, VectorType, MatrixType, Number>::get_fe_data() const
  {
    return fe_data;
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  inline Common::FEData<dim, Number> &
  Mixture_GR<dim, VectorType, MatrixType, Number>::get_fe_data()
  {
    return fe_data;
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  inline std::vector<std::shared_ptr<LocalMixture<dim, Number>>>
  Mixture_GR<dim, VectorType, MatrixType, Number>::get_local_mixture(
    const CellIteratorType<dim> &element)
  {
    return local_mixtures.get_data(element);
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  inline const dealii::CellDataStorage<CellIteratorType<dim>,
                                       LocalMixture<dim, Number>> &
  Mixture_GR<dim, VectorType, MatrixType, Number>::get_local_mixtures() const
  {
    return local_mixtures;
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  inline void
  Mixture_GR<dim, VectorType, MatrixType, Number>::
    set_time_of_prestretch_functions(const Number time) const
  {
    for (const auto &constituent_factory : constituent_factories)
      constituent_factory->set_time(time);
  }
} // namespace Mixture
