#include "MSHCMM/models/fully_coupled_model_arkode.h"

#include <deal.II/lac/sparse_direct.h>

namespace Models
{
  using namespace dealii;

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  FullyCoupledModelARKode<dim, VectorType, MatrixType, Number>::
    FullyCoupledModelARKode(
      Mixture::Mixture_GR<dim, VectorType, MatrixType, Number> &mixture,
      Diffusion::DiffusionManager<dim, VectorType, MatrixType, Number>
                                            &diffusion,
      Pathways::PathwayManager<dim, Number> &pathway_manager)
    : pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(
               mixture.get_fe_data().get_MPI_comm()) == 0))
    , computing_timer(pcout, TimerOutput::never, TimerOutput::wall_times)
    , mixture(mixture)
    , diffusion(diffusion)
    , pathway_manager(pathway_manager)
  {
    Assert(
      mixture.is_coupled() == true,
      dealii::ExcMessage(
        "When using the FullyCoupledModel, the Mixture_GR must be setup with coupled == true!"));
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  FullyCoupledModelARKode<dim, VectorType, MatrixType, Number>::
    setup_coupled_problem(
      const dealii::parallel::TriangulationBase<dim> &triangulation,
      Pathways::PathwayStorage<dim, Number>         &&pathway_storage,
      const std::shared_ptr<dealii::Function<dim>>    mixture_mass_density,
      const std::shared_ptr<dealii::Function<dim>>    initial_mass_fractions,
      const bool use_single_cell_collection)
  {
    //// PATHWAYS
    // add pathway storage to pathway_manager
    pathway_manager.setup(std::move(pathway_storage),
                          triangulation,
                          diffusion.get_fe_data().get_fe().degree,
                          diffusion.get_fe_data().get_fe().degree + 1,
                          use_single_cell_collection);
    //  distribute cells
    pathway_manager.distribute_cells();

    //// DIFFUSION
    // setup diffusion problem (set sizes of matrices, vectors, etc.)
    diffusion.setup_system();
    // compute mass matrix
    diffusion.assemble_mass_matrix();
    // interpolate initial condition
    diffusion.interpolate_initial_conditions();

    //// HCMM
    mixture.setup_system();
    mixture.setup_qp_data(mixture_mass_density, initial_mass_fractions);
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  FullyCoupledModelARKode<dim, VectorType, MatrixType, Number>::
    setup_coupled_problem(
      const std::shared_ptr<dealii::Function<dim>> mixture_mass_density,
      const std::shared_ptr<dealii::Function<dim>> initial_mass_fractions)
  {
    //// PATHWAYS
    //  distribute cells
    pathway_manager.distribute_cells();

    //// DIFFUSION
    // setup diffusion problem (set sizes of matrices, vectors, etc.)
    diffusion.setup_system();
    // compute mass matrix
    diffusion.assemble_mass_matrix();
    // interpolate initial condition
    diffusion.interpolate_initial_conditions();

    //// HCMM
    mixture.setup_system();
    mixture.setup_qp_data(mixture_mass_density, initial_mass_fractions);
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  FullyCoupledModelARKode<dim, VectorType, MatrixType, Number>::
    equilibrate_pathways(const Number total_equilibration_time,
                         const Number time_step)
  {
    (void)time_step;
    dealii::TimerOutput::Scope t(computing_timer,
                                 "CoupledProblem - Equilibrate pathways");

    pcout << "Equilibrating pathways..." << std::endl;

    // reset ODE solver final time to time
    pathway_manager.reset_ODE_solver(total_equilibration_time);

    // create alias to save some typing
    const auto &fe_data = diffusion.get_fe_data();

    auto fe_values = fe_data.make_fe_values(update_values | update_hessians |
                                            update_quadrature_points);
    const unsigned int n_q_points = fe_data.get_quadrature().size();

    auto fe_face_values =
      fe_data.make_fe_face_values(update_values | update_quadrature_points);
    const unsigned int n_q_points_face = fe_face_values.get_quadrature().size();
    // mixture FEFaceValues
    auto fe_face_values_mixture = mixture.get_fe_data().make_fe_face_values(
      update_values | update_quadrature_points);


    // values of diffusion problem
    std::vector<Vector<Number>> diffusion_values(
      n_q_points, Vector<Number>(fe_data.get_fe().components));
    // laplacians
    std::vector<Vector<Number>> diffusion_laplacians(
      n_q_points, Vector<Number>(fe_data.get_fe().components));
    // face values
    std::vector<Vector<Number>> diffusion_values_face(
      n_q_points_face, Vector<Number>(fe_data.get_fe().components));
    // allocate memory for quadrature point coordinates depending on the usage
    // of single_cell_collection
    std::vector<Point<dim>> quadrature_points(
      pathway_manager.uses_single_cell_collection() ? 1 : n_q_points);
    std::vector<Point<dim>> quadrature_points_face(
      pathway_manager.uses_single_cell_collection() ? 1 : n_q_points_face);

    // allocate memory for displacements (needed for endothelial layer)
    std::vector<Vector<Number>> displacements_face(n_q_points_face,
                                                   Vector<Number>(dim));

    // allocate memory for quadrature point data
    // diffusion parameters
    std::vector<std::shared_ptr<Diffusion::LocalDiffusionParameters<Number>>>
      local_diffusion_parameters(n_q_points);
    // cells
    std::vector<
      std::shared_ptr<Pathways::Cells::LocalCellCollection<dim, Number>>>
      local_cell_collection(n_q_points);
    // endothelial cells
    std::vector<
      std::shared_ptr<Pathways::Cells::LocalCellCollection<dim, Number>>>
      local_endothelial_cell_collection(n_q_points_face);
    // local mixture
    std::vector<std::shared_ptr<Mixture::LocalMixture<dim, Number>>>
      local_mixture(n_q_points);

    // need locally relevant solution of diffusion problem
    const auto locally_relevant_solution =
      diffusion.get_locally_relevant_solution();
    // need locally relevant solution of mixture problem
    const auto locally_relevant_solution_mixture =
      mixture.get_total_relevant_solution();

    // loop over cells
    // todo: is there a better way?
    // need to create two iterators, one for the diffusion problem and one for
    // the mixture problem
    auto element_diffusion = fe_data.get_dof_handler().begin_active();
    auto element_mixture =
      mixture.get_fe_data().get_dof_handler().begin_active();

    // for (const auto& element :
    // fe_data.get_dof_handler().active_cell_iterators())
    for (; element_diffusion != fe_data.get_dof_handler().end();
         element_diffusion++, element_mixture++)
      {
        if (element_diffusion->is_locally_owned())
          {
            // reinit fe_values to current cell
            fe_values.reinit(element_diffusion);
            // get current function values of diffusion problem
            fe_values.get_function_values(locally_relevant_solution,
                                          diffusion_values);
            fe_values.get_function_laplacians(locally_relevant_solution,
                                              diffusion_laplacians);

            // get quadrature point coordinates
            // note that if we use a single cell collection, we just create a
            // vector of size n_q_points with the coordinates of the cell center
            quadrature_points =
              pathway_manager.uses_single_cell_collection() ?
                std::vector<dealii::Point<dim>>(n_q_points,
                                                element_diffusion->center()) :
                fe_values.get_quadrature_points();

            // get local diffusion parameters
            local_diffusion_parameters =
              diffusion.get_local_diffusion_parameters_cell(element_diffusion);
            // get local cell collections
            local_cell_collection =
              pathway_manager.get_local_cell_collection(element_diffusion);
            //  get local mixture and transferable parameters
            local_mixture = mixture.get_local_mixture(element_mixture);

            // solve pathways -> results in an updated cell state based on
            // diffusion_values and HCMM state set store_baseline flag to true
            // during equilibration, time should not change, so we set it to
            // zero use empty LocalMixture
            pathway_manager.solve_pathways_on_element(quadrature_points,
                                                      0.0,
                                                      local_cell_collection,
                                                      diffusion_values,
                                                      local_mixture,
                                                      true);
            // update transferable parameters in local_mixture to get new growth
            // rates, decay times, etc.
            mixture.update_transferable_parameters(quadrature_points,
                                                   0.0,
                                                   diffusion_values,
                                                   diffusion_laplacians,
                                                   local_cell_collection,
                                                   local_mixture);
            // update local diffusion parameters based on just updated cell
            // states
            diffusion.update_local_diffusion_parameters(
              quadrature_points,
              0.0,
              diffusion_values,
              local_cell_collection,
              local_diffusion_parameters);
            // also equilibrate endothelial cells
            // loop over boundary if endothelial cells are set up and cell is at
            // the boundary
            if (pathway_manager.has_endothelial_cells() and
                element_diffusion->at_boundary())
              {
                for (const auto &face : element_diffusion->face_iterators())
                  {
                    if (!face->at_boundary())
                      continue;
                    if (face->boundary_id() ==
                        pathway_manager.get_endothelium_boundary_id())
                      {
                        // reinit fe_face_values to current cell and get current
                        // function values
                        fe_face_values.reinit(element_diffusion, face);
                        fe_face_values_mixture.reinit(element_mixture, face);
                        // get diffusion values
                        fe_face_values.get_function_values(
                          locally_relevant_solution, diffusion_values_face);
                        // get displacements
                        fe_face_values_mixture.get_function_values(
                          locally_relevant_solution_mixture,
                          displacements_face);

                        // get quadrature point coordinates
                        // note that if we use a single cell collection, we just
                        // create a vector of size n_q_points with the
                        // coordinates of the cell center
                        quadrature_points_face =
                          pathway_manager.uses_single_cell_collection() ?
                            std::vector<dealii::Point<dim>>(
                              1, element_diffusion->center(true)) :
                            fe_face_values.get_quadrature_points();
                        // get endothelial cells on the current element
                        local_endothelial_cell_collection =
                          pathway_manager.get_local_endothelial_cell_collection(
                            element_diffusion);
                        // solve pathways
                        pathway_manager.solve_pathways_on_boundary(
                          quadrature_points_face,
                          0.0,
                          local_endothelial_cell_collection,
                          diffusion_values_face,
                          displacements_face,
                          true);
                      }
                  }
              }

            // DEBUG
            //                for (unsigned int q_point = 0; q_point <
            //                local_cell_collection.size();
            //                ++q_point) {
            //                    // get local cell collection at quadrature
            //                    point auto &cell_collection_qp =
            //                    local_cell_collection[q_point];
            //
            //                    for (auto &cell :
            //                    cell_collection_qp->get_cells()) {
            //                        // get inputs
            //                        const auto &inputs =
            //                        cell.get_pathway_input();
            //                        // print inputs
            //                        std::cout << "inputs size: " <<
            //                        inputs.size() << std::endl; for (const
            //                        auto &ele : inputs)
            //                            std::cout << ele << std::endl;
            //
            //                        // get outputs
            //                        const auto outputs =
            //                        cell.get_pathway_output();
            //                        // print outputs
            //                        std::cout << "outputs size: " <<
            //                        outputs.size()  << std::endl; for (const
            //                        auto &ele : outputs)
            //                            std::cout << ele << std::endl;
            //                    }
            //                }
          }
      }

    // todo: make optional
    // Once equilibration is done, reset ODE solver to time_step used in the
    // simulation
    // pathway_manager.reset_ODE_solver(time_step);

    pcout << "Equilibrating pathways...DONE!" << std::endl;
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  FullyCoupledModelARKode<dim, VectorType, MatrixType, Number>::
    solve_pathways_and_update_diffusion_parameters(const Number       time,
                                                   const unsigned int step)
  {
    pcout << "Solving PATHWAYS at time " << time << " (step " << step << ")"
          << std::endl;
    TimerOutput::Scope t(computing_timer, "CoupledProblem - solve pathways");

    // create alias to save some typing
    const auto &fe_data_diffusion = diffusion.get_fe_data();

    // Diffusion FEValues and FEFaceValues
    auto fe_values      = fe_data_diffusion.make_fe_values(update_values |
                                                      update_quadrature_points);
    auto fe_face_values = fe_data_diffusion.make_fe_face_values(
      update_values | update_quadrature_points);
    // mixture FEFaceValues
    auto fe_face_values_mixture = mixture.get_fe_data().make_fe_face_values(
      update_values | update_quadrature_points);

    // some helper quantities
    const unsigned int n_q_points = fe_data_diffusion.get_quadrature().size();
    const unsigned int n_q_points_face = fe_face_values.get_quadrature().size();

    // allocate memory for diffusion values
    std::vector<Vector<Number>> diffusion_values(
      n_q_points, Vector<Number>(fe_data_diffusion.get_fe().components));
    std::vector<Vector<Number>> diffusion_values_face(
      n_q_points_face, Vector<Number>(fe_data_diffusion.get_fe().components));

    // todo: also get diffusion laplacians so that diffusion of constituents can
    // be included
    //  this is then passed to mixture.update_transferable_parameters() which
    //  also needs to be modified to take that additional argument. In the
    //  fully_coupled_model_test, the transformer then also needs to be adjusted
    //  to include: transferable_parameters.mass_production =
    //            k_0_SMC * (1.0 + delta_psi_1) * diffusion_values[0]
    //            + D_constituent * diffusion_laplacian[constituent_index] BUT
    //            ONLY IF MASS INFLUX
    //  (need to check which sign that corresponds to...I think + ?)
    //  IMPORTANT: This only makes sense if quadratic elements are used!!!
    //  Linear elements do NOT have a second derivative!

    // allocate memory for displacements (needed for endothelial layer)
    std::vector<Vector<Number>> displacements_face(n_q_points_face,
                                                   Vector<Number>(dim));

    // allocate memory for quadrature point coordinates depending on the usage
    // of single_cell_collection
    std::vector<Point<dim>> quadrature_points(
      pathway_manager.uses_single_cell_collection() ? 1 : n_q_points);
    std::vector<Point<dim>> quadrature_points_face(
      pathway_manager.uses_single_cell_collection() ? 1 : n_q_points_face);
    //// allocate memory for
    // - diffusion parameters
    std::vector<std::shared_ptr<Diffusion::LocalDiffusionParameters<Number>>>
      local_diffusion_parameters(n_q_points);
    // - cells
    std::vector<
      std::shared_ptr<Pathways::Cells::LocalCellCollection<dim, Number>>>
      local_cell_collection(
        pathway_manager.uses_single_cell_collection() ? 1 : n_q_points);
    // - endothelial cells
    std::vector<
      std::shared_ptr<Pathways::Cells::LocalCellCollection<dim, Number>>>
      local_endothelial_cell_collection(
        pathway_manager.uses_single_cell_collection() ? 1 : n_q_points_face);
    // - mixture
    std::vector<std::shared_ptr<Mixture::LocalMixture<dim, Number>>>
      local_mixture(n_q_points);

    // need locally relevant solution of diffusion problem
    const auto locally_relevant_solution =
      diffusion.get_locally_relevant_solution();

    // loop over cells
    // todo: is there a better way?
    // need to create two iterators, one for the diffusion problem and one for
    // the mixture problem
    auto element_diffusion = fe_data_diffusion.get_dof_handler().begin_active();
    auto element_mixture =
      mixture.get_fe_data().get_dof_handler().begin_active();

    // for (const auto& element :
    // fe_data_diffusion.get_dof_handler().active_cell_iterators())
    for (; element_diffusion != fe_data_diffusion.get_dof_handler().end();
         element_diffusion++, element_mixture++)
      {
        if (element_diffusion->is_locally_owned())
          {
            // reinit fe_values to current cell and get current function values
            fe_values.reinit(element_diffusion);
            // get function values
            fe_values.get_function_values(locally_relevant_solution,
                                          diffusion_values);

            // get quadrature point coordinates
            // note that if we use a single cell collection, we just create a
            // vector of size n_q_points with the coordinates of the cell center
            quadrature_points =
              pathway_manager.uses_single_cell_collection() ?
                std::vector<dealii::Point<dim>>(
                  1, element_diffusion->center(true /*repect_manifold*/)) :
                fe_values.get_quadrature_points();

            // get local diffusion parameters
            local_diffusion_parameters =
              diffusion.get_local_diffusion_parameters_cell(element_diffusion);
            // get local mixture and get/update transferable parameters
            local_mixture = mixture.get_local_mixture(element_mixture);
            // get local cell collection
            local_cell_collection =
              pathway_manager.get_local_cell_collection(element_diffusion);

            // solve pathways -> results in an updated cell state based on
            // diffusion_values and HCMM state
            pathway_manager.solve_pathways_on_element(quadrature_points,
                                                      time,
                                                      local_cell_collection,
                                                      diffusion_values,
                                                      local_mixture);

            // update local diffusion parameters based on just updated cell
            // states
            diffusion.update_local_diffusion_parameters(
              quadrature_points,
              time,
              diffusion_values,
              local_cell_collection,
              local_diffusion_parameters);

            // loop over boundary id there are endothelial cells and cell is at
            // the boundary
            if (pathway_manager.has_endothelial_cells() and
                element_diffusion->at_boundary())
              {
                for (const auto &face : element_diffusion->face_iterators())
                  {
                    if (!face->at_boundary())
                      continue;
                    if (face->boundary_id() ==
                        pathway_manager.get_endothelium_boundary_id())
                      {
                        // reinit fe_face_values to current cell and get current
                        // function values
                        fe_face_values.reinit(element_diffusion, face);
                        fe_face_values_mixture.reinit(element_mixture, face);

                        // get diffusion values
                        fe_face_values.get_function_values(
                          locally_relevant_solution, diffusion_values_face);
                        // get displacements
                        fe_face_values_mixture.get_function_values(
                          mixture.get_total_relevant_solution(),
                          displacements_face);
                        // get quadrature point coordinates
                        // note that if we use a single cell collection, we just
                        // create a vector of size n_q_points with the
                        // coordinates of the cell center with respect_manifold
                        // = true
                        quadrature_points_face =
                          pathway_manager.uses_single_cell_collection() ?
                            std::vector<dealii::Point<dim>>(
                              1,
                              element_diffusion->center(
                                true /*repect_manifold*/)) :
                            fe_face_values.get_quadrature_points();
                        // get endothelial cells on the current element
                        local_endothelial_cell_collection =
                          pathway_manager.get_local_endothelial_cell_collection(
                            element_diffusion);
                        // solve pathways
                        pathway_manager.solve_pathways_on_boundary(
                          quadrature_points_face,
                          time,
                          local_endothelial_cell_collection,
                          diffusion_values_face,
                          displacements_face);
                      }
                  }
              }
          }
      }
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  FullyCoupledModelARKode<dim, VectorType, MatrixType, Number>::
    setup_time_stepper(
      const typename SUNDIALS::ARKode<VectorType>::AdditionalData
                  &time_stepper_data,
      const Number max_dt,
      const Number rabstol,
      const int    predictor_method)
  {
    this->time_stepper_settings = time_stepper_data;

    time_stepper = std::make_unique<SUNDIALS::ARKode<VectorType>>(
      time_stepper_data, diffusion.get_fe_data().get_MPI_comm());

    // some custom setup, set maximum time step size. Important to set
    // ResTolerance!!!
    time_stepper->custom_setup = [&, max_dt](void *arkode_mem) {
      // can be important when time steps are small, e.g., when pathways should
      // be solved every 15 minutes, i.e., every 0.25 hours
      ARKStepSetMaxStep(arkode_mem, max_dt);
      // set predictor
      ARKStepSetPredictorMethod(arkode_mem, predictor_method);
      // set scalar absolute residual tolerance, see ARKode 3.4.2.2.2.1. General
      // advice on the choice of tolerances
      ARKStepResStolerance(arkode_mem, rabstol);
    };

    // implicit function
    time_stepper->implicit_function =
      [&](const double t, const VectorType &y, VectorType &ydot) -> int {
      (void)t;
      // assemble degradation part
      diffusion.assemble_K(y);
      // add laplace part
      diffusion.K.add(1.0, diffusion.laplace_matrix);
      // multiply
      diffusion.K.vmult(ydot, y);

      // assemble system rhs
      diffusion.assemble_source_and_boundary_terms(
        t, pathway_manager.uses_single_cell_collection());
      // add endothelial cell contribution
      if (pathway_manager.has_endothelial_cells())
        {
          const auto boundary_values =
            diffusion.assemble_endothelial_contribution(pathway_manager);
          // todo: do not hardcode parameters
          const double       outflux_fraction = 0.9;
          const unsigned int outflux_bc       = 1;
          diffusion.assemble_outlfux_boundary(boundary_values,
                                              outflux_fraction,
                                              outflux_bc);
        }

      // add source and boundary terms
      ydot += diffusion.system_rhs;

      return 0;
    };

    //    // explicit function
    //    time_stepper->explicit_function = [&](const double t, const
    //    VectorType& y, VectorType& explicit_f) -> int {
    //      (void) y;
    //
    //      // assemble system rhs
    //      diffusion.assemble_source_and_boundary_terms(t, false);
    //      // add source and boundary terms
    //      explicit_f = diffusion.system_rhs;
    //
    //      return 0;
    //    };

    // jacobian times vector
    time_stepper->jacobian_times_vector = [&](const VectorType &v,
                                              VectorType       &Jv,
                                              double            t,
                                              const VectorType &y,
                                              const VectorType &fy) -> int {
      (void)t;
      (void)y;
      (void)fy;

      diffusion.jacobian.vmult(Jv, v);

      return 0;
    };

    // assemble jacobian
    time_stepper->jacobian_times_setup =
      [&](double t, const VectorType &y, const VectorType &fy) -> int {
      (void)t;
      (void)fy;

      // assemble jacobian part
      diffusion.assemble_jacobian(y);
      // add laplace to jacobian matrix
      diffusion.jacobian.add(1.0, diffusion.laplace_matrix);

      return 0;
    };

    // mass times vector
    time_stepper->mass_times_vector =
      [&](const double t, const VectorType &v, VectorType &Mv) -> int {
      (void)t;

      diffusion.mass_matrix.vmult(Mv, v);

      return 0;
    };

    // apply jacobian preconditioner
    time_stepper->jacobian_preconditioner_solve = [&](double            t,
                                                      const VectorType &y,
                                                      const VectorType &fy,
                                                      const VectorType &r,
                                                      VectorType       &z,
                                                      double            gamma,
                                                      double            tol,
                                                      int lr) -> int {
      (void)t;
      (void)y;
      (void)fy;
      (void)gamma;
      (void)tol;
      (void)lr;
      // apply preconditioner
      diffusion.preconditioner->vmult(z, r);

      return 0;
    };

    // setup jacobian preconditioner
    // setup preconditioner
    TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;

    // AMG settings
    additional_data.higher_order_elements =
      diffusion.get_fe_data().get_fe().degree > 1;
    additional_data.elliptic              = true;
    additional_data.n_cycles              = 1;
    additional_data.w_cycle               = false;
    additional_data.output_details        = false;
    additional_data.smoother_sweeps       = 2;
    additional_data.aggregation_threshold = 1e-4;

    diffusion.preconditioner =
      std::make_unique<TrilinosWrappers::PreconditionAMG>();
    diffusion.preconditioner->initialize(diffusion.system_matrix,
                                         additional_data);

    time_stepper->jacobian_preconditioner_setup = [&](double            t,
                                                      const VectorType &y,
                                                      const VectorType &fy,
                                                      int               jok,
                                                      int              &jcur,
                                                      double gamma) -> int {
      (void)t;
      (void)y;
      (void)fy;
      (void)jok;
      (void)jcur;

      // build matrix
      diffusion.system_matrix.copy_from(diffusion.mass_matrix);
      diffusion.system_matrix.add(-gamma, diffusion.laplace_matrix);
      diffusion.system_matrix.add(-gamma, diffusion.jacobian);
      // rebuild preconditioner
      diffusion.preconditioner->reinit();

      return 0;
    };

    // apply mass preconditioner
    time_stepper->mass_preconditioner_solve = [&](double            t,
                                                  const VectorType &r,
                                                  VectorType       &z,
                                                  double            tol,
                                                  int               lr) -> int {
      (void)t;
      (void)tol;
      (void)lr;

      diffusion.preconditioner_mass->vmult(z, r);

      return 0;
    };

    // setup mass preconditioner
    diffusion.preconditioner_mass =
      std::make_unique<TrilinosWrappers::PreconditionAMG>();

    // initialize preconditioner
    diffusion.preconditioner_mass->initialize(diffusion.mass_matrix,
                                              additional_data);

    time_stepper->mass_preconditioner_setup = [&](double t) -> int {
      (void)t;

      diffusion.preconditioner_mass->reinit();

      return 0;
    };


    // solve linearized system
    const auto solve_function =
      [&](SUNDIALS::SundialsOperator<VectorType>       &op,
          SUNDIALS::SundialsPreconditioner<VectorType> &prec,
          VectorType                                   &x,
          const VectorType                             &b,
          double                                        tol) -> int {
      TimerOutput::Scope t(computing_timer,
                           "DiffusionManager - Solve linearized system");
      SolverControl      control(1000, tol);

      SolverGMRES<VectorType> solver(control);
      solver.solve(op, x, b, prec);

      //      pcout << "iter: " << control.last_step() << " res: " <<
      //      control.last_value() << std::endl;

      iteration_tracker.lin_iters_jac += control.last_step();

      return 0;
    };

    time_stepper->solve_linearized_system = solve_function;


    // solve linearized system
    const auto solve_function_mass =
      [&](SUNDIALS::SundialsOperator<VectorType>       &op,
          SUNDIALS::SundialsPreconditioner<VectorType> &prec,
          VectorType                                   &x,
          const VectorType                             &b,
          double                                        tol) -> int {
      TimerOutput::Scope t(computing_timer, "DiffusionManager - Solve mass");
      SolverControl      control(100, tol);

      SolverCG<VectorType> solver(control);
      solver.solve(op, x, b, prec);

      //      pcout << "iter mass: " << control.last_step() << " res: " <<
      //      control.last_value() << std::endl;

      return 0;
    };

    time_stepper->solve_mass = solve_function_mass;
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  FullyCoupledModelARKode<dim, VectorType, MatrixType, Number>::
    update_transferable_parameters_mixture(const Number       time,
                                           const unsigned int step)
  {
    (void)step;
    TimerOutput::Scope t(
      computing_timer,
      "CoupledProblem - update transferable parameters of mixture");

    // create alias to save some typing
    const auto &fe_data_diffusion = diffusion.get_fe_data();

    // Diffusion FEValues and FEFaceValues
    auto fe_values = fe_data_diffusion.make_fe_values(
      update_values | update_hessians | update_quadrature_points);

    // some helper quantities
    const unsigned int n_q_points = fe_data_diffusion.get_quadrature().size();

    // allocate memory for diffusion values
    std::vector<Vector<Number>> diffusion_values(
      n_q_points, Vector<Number>(fe_data_diffusion.get_fe().components));
    // laplacians
    std::vector<Vector<Number>> diffusion_laplacians(
      n_q_points, Vector<Number>(fe_data_diffusion.get_fe().components));
    // todo: also get diffusion laplacians so that diffusion of constituents can
    // be included
    //  this is then passed to mixture.update_transferable_parameters() which
    //  also needs to be modified to take that additional argument. In the
    //  fully_coupled_model_test, the transformer then also needs to be adjusted
    //  to include: transferable_parameters.mass_production =
    //            k_0_SMC * (1.0 + delta_psi_1) * diffusion_values[0]
    //            + D_constituent * diffusion_laplacian[constituent_index] BUT
    //            ONLY IF MASS INFLUX
    //  (need to check which sign that corresponds to...I think + ?)
    //  IMPORTANT: This only makes sense if quadratic elements are used!!!
    //  Linear elements do NOT have a second derivative!

    // allocate memory for quadrature point coordinates depending on the usage
    // of single_cell_collection
    std::vector<Point<dim>> quadrature_points(
      pathway_manager.uses_single_cell_collection() ? 1 : n_q_points);
    //// allocate memory for
    // - cells
    std::vector<
      std::shared_ptr<Pathways::Cells::LocalCellCollection<dim, Number>>>
      local_cell_collection(
        pathway_manager.uses_single_cell_collection() ? 1 : n_q_points);
    // - mixture
    std::vector<std::shared_ptr<Mixture::LocalMixture<dim, Number>>>
      local_mixture(n_q_points);

    // need locally relevant solution of diffusion problem
    const auto locally_relevant_solution =
      diffusion.get_locally_relevant_solution();

    // loop over cells
    for (const auto &element :
         fe_data_diffusion.get_dof_handler().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            // reinit fe_values to current cell and get current function values
            fe_values.reinit(element);
            // get current values of diffusion problem
            fe_values.get_function_values(locally_relevant_solution,
                                          diffusion_values);
            fe_values.get_function_laplacians(locally_relevant_solution,
                                              diffusion_laplacians);

            // get quadrature point coordinates
            // note that if we use a single cell collection, we just create a
            // vector of size n_q_points with the coordinates of the cell center
            // todo: add respect_manifold = true to element->center()?
            quadrature_points =
              pathway_manager.uses_single_cell_collection() ?
                std::vector<dealii::Point<dim>>(
                  1, element->center(true /*repect_manifold*/)) :
                fe_values.get_quadrature_points();

            // get local mixture and get/update transferable parameters
            local_mixture = mixture.get_local_mixture(element);
            // get local cell collection
            local_cell_collection =
              pathway_manager.get_local_cell_collection(element);

            // update transferable parameters in local_mixture to get new growth
            // rates, decay times, etc.
            mixture.update_transferable_parameters(quadrature_points,
                                                   time,
                                                   diffusion_values,
                                                   diffusion_laplacians,
                                                   local_cell_collection,
                                                   local_mixture);
          }
      }
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  FullyCoupledModelARKode<dim, VectorType, MatrixType, Number>::
    update_GR_mixture()
  {}

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  FullyCoupledModelARKode<dim, VectorType, MatrixType, Number>::run(
    const unsigned int n_time_steps,
    const Number       initial_time,
    const Number       final_time,
    const unsigned int output_diffusion_every_x_steps,
    const Number       pathway_equilibration_time,
    const unsigned int solve_pathways_every_x_steps,
    const unsigned int output_pathways_every_x_steps,
    const unsigned int solve_HCMM_every_x_steps,
    const bool         active_GR)
  {
    // compute time step size
    const double time_step =
      (final_time - initial_time) / static_cast<double>(n_time_steps);
    // time object
    DiscreteTime time(initial_time, final_time, time_step);

    // solve mixture once before starting the time loop so that all the stresses
    // are set properly to their homeostatic values
    // mixture.solve_step_with_NOX(0, 0.0, false);
    // Note that the second argument is the total time the pathways should be
    // evolved/solved, therefore, it needs to be multiplied by
    // solve_pathways_every_x_steps in order to ensure the correct time in case
    // the pathways are not solved at every step of the diffusion problem
    equilibrate_pathways(pathway_equilibration_time,
                         solve_pathways_every_x_steps * time_step);
    pathway_manager.write_output(0, 0.0);
    // output initial diffusion state
    diffusion.output_results(0.0, 0);
    // update transferable parameters
    update_transferable_parameters_mixture(0.0, 0);
    // write initial mixture state
    mixture.output_results(0);

    Assert(time_stepper,
           dealii::ExcMessage(
             "ARKode was not initialized! Call setup_time_stepper() first!"));

    // start time loop
    while (!time.is_at_end())
      {
        time.advance_time();

        const auto current_step = time.get_step_number();
        const auto current_time = time.get_current_time();

        //// solve mixture
        if (current_step % solve_HCMM_every_x_steps == 0)
          {
            // solve step
            mixture.solve_step_with_NOX(current_step, current_time, false);
          }
        //// solve pathways
        if (current_step % solve_pathways_every_x_steps == 0)
          solve_pathways_and_update_diffusion_parameters(current_time,
                                                         current_step);

        //// solve diffusion
        pcout << "Solving DIFFUSION at time " << current_time << " (step "
              << current_step << ")" << std::endl;
        // reset iteration tracker
        iteration_tracker.reset_stats();
        // solve step
        time_stepper->solve_ode_incrementally(diffusion.solution,
                                              current_time,
                                              false);
        // get time stepper data
        iteration_tracker.update_stats(time_stepper->get_arkode_memory());
        // print stats
        iteration_tracker.print_stats(pcout);

        // update G&R in mixture once pathways and diffusion have been solved
        if (active_GR and current_step % solve_HCMM_every_x_steps == 0)
          {
            // first, update transferable parameters of mixture constituents
            // with values from diffusion and pathways
            update_transferable_parameters_mixture(current_time, current_step);
            // then, update G&R
            mixture.update_GR();
            // write output
            mixture.output_results(current_step);
          }

        //// diffusion output
        if (current_step % output_diffusion_every_x_steps == 0)
          diffusion.output_results(current_time, current_step);
        //// pathway output
        if (current_step % output_pathways_every_x_steps == 0)
          pathway_manager.write_output(current_step, current_time);
      }

    // when done, print some timing statistics
    pathway_manager.print_timer_stats(true);
    diffusion.print_timer_stats(true);
    mixture.print_timer_stats(true);

    pcout << "CoupledProblem timings";
    computing_timer.print_summary();
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  Diffusion::DiffusionManager<dim, VectorType, MatrixType, Number> &
  FullyCoupledModelARKode<dim, VectorType, MatrixType, Number>::get_diffusion()
  {
    return diffusion;
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  Mixture::Mixture_GR<dim, VectorType, MatrixType, Number> &
  FullyCoupledModelARKode<dim, VectorType, MatrixType, Number>::get_mixture()
  {
    return mixture;
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  Pathways::PathwayManager<dim, Number> &
  FullyCoupledModelARKode<dim, VectorType, MatrixType, Number>::
    get_pathway_manager()
  {
    return pathway_manager;
  }

  // instantiations
  template class FullyCoupledModelARKode<3,
                                         dealii::TrilinosWrappers::MPI::Vector,
                                         dealii::TrilinosWrappers::SparseMatrix,
                                         double>;
} // namespace Models