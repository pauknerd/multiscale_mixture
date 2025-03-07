#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/base/function.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <MSHCMM/diffusion/diffusion_manager.h>
#include <MSHCMM/mixture/growth_and_remodeling/mixture_G&R.h>
#include <MSHCMM/pathways/pathway_manager.h>

#include <iostream>
#include <memory>

namespace Models
{

  /**
   * Fully coupled model.
   */
  template <int dim,
            typename VectorType,
            typename MatrixType,
            typename Number = double>
  class FullyCoupledModelARKode
  {
  public:
    /**
     * Constructor. Assumes Mixture_GR has been setup with coupled == true.
     *
     * @param mixture
     * @param diffusion
     * @param pathway_manager
     */
    FullyCoupledModelARKode(
      Mixture::Mixture_GR<dim, VectorType, MatrixType, Number> &mixture,
      Diffusion::DiffusionManager<dim, VectorType, MatrixType, Number>
                                            &diffusion,
      Pathways::PathwayManager<dim, Number> &pathway_manager);

    void
    setup_coupled_problem(
      const dealii::parallel::TriangulationBase<dim> &triangulation,
      Pathways::PathwayStorage<dim, Number>         &&pathway_storage,
      const std::shared_ptr<dealii::Function<dim>>    mixture_mass_density,
      const std::shared_ptr<dealii::Function<dim>>    initial_mass_fractions,
      const bool use_single_cell_collection = false);

    /**
     * Bundles some setup calls of the different sub-problems. Assumes that the
     * setup method of PathwayManager was already called to transfer the
     * PathwayStorage.
     *
     * @param mixture_mass_density
     * @param initial_mass_fractions
     */
    void
    setup_coupled_problem(
      const std::shared_ptr<dealii::Function<dim>> mixture_mass_density,
      const std::shared_ptr<dealii::Function<dim>> initial_mass_fractions);

    /**
     * @brief equilibrate pathways for time t.
     *
     * Assumes HCMM has been solved before so that stress inputs are available
     * in transferable parameters! After equilibration is done, rest ODE solver
     * to time step used in the simulation.
     *
     * NOTE: During equilibration, the time parameter of the transformers is set
     * 0.0! This is done because the values during pathway equilibration
     * shouldn't change. That means the pathways are equilibrated to function
     * values at time 0.0.
     *
     * @param total_equilibration_time
     * @param time_step
     */
    void
    equilibrate_pathways(const Number total_equilibration_time,
                         const Number time_step);

    void
    solve_pathways_and_update_diffusion_parameters(const Number       time,
                                                   const unsigned int step);

    void
    update_transferable_parameters_mixture(const Number       time,
                                           const unsigned int step);

    void
    update_GR_mixture();

    void
    setup_time_stepper(
      const typename SUNDIALS::ARKode<VectorType>::AdditionalData
                  &time_stepper_data,
      const Number max_dt,
      const Number rabstol,
      const int    predictor_method = 0);

    void
    run(const unsigned int n_time_steps,
        const Number       initial_time,
        const Number       final_time,
        const unsigned int output_diffusion_every_x_steps,
        const Number       pathway_equilibration_time,
        const unsigned int solve_pathways_every_x_steps,
        const unsigned int output_pathways_every_x_steps,
        const unsigned int solve_HCMM_every_x_steps,
        const bool         active_GR);

    Diffusion::DiffusionManager<dim, VectorType, MatrixType, Number> &
    get_diffusion();

    Mixture::Mixture_GR<dim, VectorType, MatrixType, Number> &
    get_mixture();

    Pathways::PathwayManager<dim, Number> &
    get_pathway_manager();

  private:
    dealii::ConditionalOStream pcout;
    dealii::TimerOutput        computing_timer;

    // homogenized constrained mixture
    Mixture::Mixture_GR<dim, VectorType, MatrixType, Number> &mixture;
    // diffusion problem
    Diffusion::DiffusionManager<dim, VectorType, MatrixType, Number> &diffusion;
    // pathways
    Pathways::PathwayManager<dim, Number> &pathway_manager;

    // time stepping
    typename SUNDIALS::ARKode<VectorType>::AdditionalData time_stepper_settings;
    std::unique_ptr<SUNDIALS::ARKode<VectorType>>         time_stepper;
    // iteration tracker for time stepper
    IterationTracker iteration_tracker;
  };
} // namespace Models
