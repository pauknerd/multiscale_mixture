#include <deal.II/distributed/cell_data_transfer.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_selector.h>

#include <deal.II/physics/transformations.h>

#include <MSHCMM/mixture/growth_and_remodeling/mixture_G&R.h>
#include <MSHCMM/utilities/helpers.h>

namespace Mixture
{
  template <int dim, typename VectorType, typename MatrixType, typename Number>
  Mixture_GR<dim, VectorType, MatrixType, Number>::Mixture_GR(
    Common::FEData<dim, Number> &&fe_data,
    std::unique_ptr<GrowthStrategies::GrowthStrategyBase<dim, Number>>
      growth_strategy,
    std::unique_ptr<PrestretchStrategies::PrestretchStrategyBase<dim, Number>>
      prestretch_strategy,
    std::vector<
      std::unique_ptr<Constituents::ConstituentFactoryBase<dim, Number>>>
                                                   constituent_factories,
    const Common::BoundaryDescriptor<dim, Number> &new_boundary_descriptor,
    const bool                                     coupled)
    : mpi_communicator(fe_data.get_dof_handler().get_communicator())
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(pcout,
                      dealii::TimerOutput::never,
                      dealii::TimerOutput::wall_times)
    , fe_data(std::move(fe_data))
    , growth_strategy(std::move(growth_strategy))
    , prestretch_strategy(std::move(prestretch_strategy))
    , constituent_factories(std::move(constituent_factories))
    , boundary_descriptor(new_boundary_descriptor)
    , coupled(coupled)
    , u_fe(0)
  {}

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  Mixture_GR<dim, VectorType, MatrixType, Number>::Mixture_GR(
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
    const bool                                     coupled)
    : mpi_communicator(triangulation.get_communicator())
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(pcout,
                      dealii::TimerOutput::never,
                      dealii::TimerOutput::wall_times)
    , fe_data(triangulation, fe_degree, quad_degree, dim)
    , growth_strategy(std::move(growth_strategy))
    , prestretch_strategy(std::move(prestretch_strategy))
    , constituent_factories(std::move(constituent_factories))
    , boundary_descriptor(new_boundary_descriptor)
    , coupled(coupled)
    , u_fe(0)
  {}

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  Mixture_GR<dim, VectorType, MatrixType, Number>::setup_system()
  {
    // print some statistics
    pcout << "Number of global active elements in Mixture: "
          << fe_data.get_triangulation().n_global_active_cells() << std::endl;

    pcout << "Number of global dofs in Mixture: "
          << fe_data.get_dof_handler().n_dofs() << std::endl;

    // initialize system vectors and matrices
    fe_data.initialize_matrix(system_state.system_matrix);
    fe_data.initialize_locally_owned_vector(system_state.system_rhs);

    fe_data.initialize_locally_owned_vector(system_state.solution);
    fe_data.initialize_locally_owned_vector(system_state.old_solution);
    fe_data.initialize_locally_owned_vector(system_state.solution_delta);
    // initialize predictor
    fe_data.initialize_locally_owned_vector(system_state.predictor);

    // compute reference volume
    reference_volume = dealii::GridTools::volume(fe_data.get_triangulation(),
                                                 fe_data.get_mapping());

    // initialize zero constraints object
    boundary_descriptor.build_dirichlet_constraints(
      fe_data.get_dof_handler(), fe_data.get_zero_constraints(), true);

    system_state.old_solution = system_state.solution;
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  Mixture_GR<dim, VectorType, MatrixType, Number>::setup_qp_data(
    const std::shared_ptr<dealii::Function<dim>> initial_reference_density,
    const std::shared_ptr<dealii::Function<dim>> initial_mass_fractions)
  {
    // create a temporary map
    std::map<unsigned int, std::shared_ptr<dealii::Function<dim>>> temp_map;
    // assumes that material id is 0!
    temp_map.emplace(/*material_id*/ 0, initial_mass_fractions);

    // call other setup_qp_data function
    setup_qp_data(initial_reference_density, temp_map);
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  Mixture_GR<dim, VectorType, MatrixType, Number>::setup_qp_data(
    const std::shared_ptr<dealii::Function<dim>> initial_reference_density,
    const std::map<unsigned int, std::shared_ptr<dealii::Function<dim>>>
      &initial_mass_fractions)
  {
    dealii::TimerOutput::Scope timer_section(
      computing_timer, "Mixture - Setup quadrature point data");

    pcout << "Setting up quadrature point data..." << std::endl;
    // setup quadrature point data of constituents
    const auto n_q_points = fe_data.get_quadrature().size();
    // create filter to extract locally owned cells
    using CellFilter = dealii::FilteredIterator<
      typename dealii::Triangulation<dim>::active_cell_iterator>;

    const auto &tria = fe_data.get_triangulation();

    local_mixtures.initialize(
      CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
                 tria.begin_active()),
      CellFilter(dealii::IteratorFilters::LocallyOwnedCell(), tria.end()),
      n_q_points);

    // vectors holding the evaluated function values
    std::vector<Number> initial_reference_density_values(n_q_points);

    // create FEValues to get locations of quadrature points. Needed to evaluate
    // functions such as initial_reference_density, initial_mass_fractions, and
    // create_constituent() method of factories
    auto fe_values = fe_data.make_fe_values(dealii::update_quadrature_points);

    for (const auto &element :
         fe_data.get_dof_handler().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            // get LocalMixture data, empty so far
            const std::vector<std::shared_ptr<LocalMixture<dim, Number>>>
              qp_data = local_mixtures.get_data(element);
            Assert(qp_data.size() == n_q_points, dealii::ExcInternalError());

            // reinit fe_values to current element
            fe_values.reinit(element);

            // get material_id of element
            const auto material_id = element->material_id();
            // get initial mass fraction function associated with that
            // material_id
            const auto &initial_mass_fraction_function =
              initial_mass_fractions.at(material_id);
            // create vector to store evaluated function
            std::vector<dealii::Vector<Number>> initial_mass_fraction_values(
              n_q_points,
              dealii::Vector<Number>(
                initial_mass_fraction_function->n_components));

            // get coordinates of quadrature points
            const auto &quadrature_points = fe_values.get_quadrature_points();

            // evaluate functions of reference density and mass fractions at
            // quadrature points
            initial_reference_density->value_list(
              quadrature_points, initial_reference_density_values);
            initial_mass_fraction_function->vector_value_list(
              quadrature_points, initial_mass_fraction_values);

            // loop over quadrature points need to check location of quadrature
            // point to make initial_reference_density, initial_mass_fractions,
            // and material parameters space dependent need FEValues with
            // update_quadrature_points flags
            for (const auto q_point : fe_values.quadrature_point_indices())
              {
                // create constituents based on stored factories
                std::vector<
                  std::unique_ptr<Constituents::ConstituentBase<dim, Number>>>
                  constituents;
                // loop over all factories and create associated constituent if
                // present on current element based on material_id
                for (const auto &factory : constituent_factories)
                  {
                    // check if constituent is associated with current element
                    if (factory->check_material_id(element->material_id()))
                      constituents.push_back(factory->create_constituent(
                        quadrature_points[q_point]));
                  }

                // helper function to convert dealii::Vector to std::vector
                auto to_std_vector =
                  [](const dealii::Vector<Number> &vector_in) {
                    std::vector<Number> vector_out;
                    std::copy(vector_in.begin(),
                              vector_in.end(),
                              std::back_inserter(vector_out));
                    return vector_out;
                  };

                Assert(
                  initial_mass_fraction_function->n_components ==
                    constituents.size(),
                  dealii::ExcMessage(
                    "Number of components of initial mass fractions function "
                    "does not match number of constituents for elements with"
                    "material id " +
                    std::to_string(material_id) + "!"));

                // transfer just created constituents
                qp_data[q_point]->setup_local_data(
                  quadrature_points[q_point],
                  initial_reference_density_values[q_point],
                  to_std_vector(initial_mass_fraction_values[q_point]),
                  std::move(constituents));
              }
          }
      }

    // TODO: maybe this can be improved and included in the loop above
    // needed to set initial stress and tangent in local mixtures
    update_material_data();
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  Mixture_GR<dim, VectorType, MatrixType, Number>::assemble_matrix(
    const VectorType &total_solution)
  {
    dealii::TimerOutput::Scope timer_section(computing_timer,
                                             "Mixture - Assemble matrix");

    // need locally relevant solution
    VectorType total_relevant_solution;
    fe_data.initialize_locally_relevant_vector(total_relevant_solution);
    total_relevant_solution = total_solution;
    total_relevant_solution.update_ghost_values();

    // reset system matrix
    system_state.system_matrix = 0.0;

    // create FEValues objects
    auto fe_values =
      fe_data.make_fe_values(dealii::update_values | dealii::update_gradients |
                             dealii::update_JxW_values);
    auto fe_face_values = fe_data.make_fe_face_values(
      dealii::update_values | dealii::update_normal_vectors |
      dealii::update_gradients | dealii::update_quadrature_points |
      dealii::update_JxW_values);

    const auto &fe = fe_data.get_fe();

    const unsigned int dofs_per_cell = fe_data.get_fe().n_dofs_per_cell();
    const unsigned int n_q_points    = fe_data.get_quadrature().size();

    dealii::FullMatrix<Number> cell_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);

    // need deformation gradients on faces
    std::vector<dealii::Tensor<2, dim, Number>> displacement_gradients(
      fe_face_values.get_quadrature().size());
    std::vector<dealii::Tensor<2, dim, Number>> F_face(
      fe_face_values.get_quadrature().size());

    // some temporary structures used during assembly
    std::vector<std::vector<Number>> Nx(n_q_points,
                                        std::vector<Number>(dofs_per_cell));

    std::vector<std::vector<dealii::Tensor<2, dim, Number>>> grad_Nx(
      n_q_points, std::vector<dealii::Tensor<2, dim>>(dofs_per_cell));
    std::vector<std::vector<dealii::SymmetricTensor<2, dim, Number>>>
      symm_grad_Nx(n_q_points,
                   std::vector<dealii::SymmetricTensor<2, dim>>(dofs_per_cell));

    // unit symmetric tensor
    const auto I = dealii::unit_symmetric_tensor<dim>();

    //// EVALUATE loop over elements
    for (const auto &element :
         fe_data.get_dof_handler().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            fe_values.reinit(element);
            // get local dof indices
            element->get_dof_indices(local_dof_indices);

            // reset element matrix
            cell_matrix = 0.0;

            // get local mixtures of current element
            auto local_mixture = local_mixtures.get_data(element);

            // fill values
            for (const auto q_point : fe_values.quadrature_point_indices())
              {
                // get deformation gradient at quadrature point
                const auto F =
                  local_mixture[q_point]->get_mixture_deformation_gradient();
                // pcout << "F: " << F << std::endl;
                for (const unsigned int k : fe_values.dof_indices())
                  {
                    const unsigned int k_group =
                      fe.system_to_base_index(k).first.first;

                    if (k_group == fe_data.u_dof)
                      {
                        grad_Nx[q_point][k] =
                          fe_values[u_fe].gradient(k, q_point);
                        // NOTE: important to include transpose(F) for proper
                        // linearization!!! symm_grad_Nx corresponds to deltaE
                        // in Holzapfel (see page 394)
                        symm_grad_Nx[q_point][k] = dealii::symmetrize(
                          transpose(F) * grad_Nx[q_point][k]);
                      }
                    else
                      Assert(k_group <= fe_data.u_dof,
                             dealii::ExcInternalError());
                  }
              }

            // start assembly loop
            for (const auto q_point : fe_values.quadrature_point_indices())
              {
                // get stress and tangent matrices at current quadrature point
                // from local mixture, note that these are already the
                // mass-averaged quantities.
                const dealii::Tensor<2, dim> stress_ns =
                  local_mixture[q_point]->get_mixture_stress();
                const dealii::SymmetricTensor<4, dim> tangent =
                  local_mixture[q_point]->get_mixture_tangent();

                // temporary variables, supposed to speed up assembly process
                // (step 44)
                dealii::SymmetricTensor<2, dim> symm_grad_Nx_i_x_tangent;
                dealii::Tensor<1, dim>          grad_Nx_i_comp_i_x_stress;

                // some abbreviations (based on step 44)
                const std::vector<dealii::SymmetricTensor<2, dim>>
                  &symm_grad_Nx_ = symm_grad_Nx[q_point];
                const std::vector<dealii::Tensor<2, dim>> &grad_Nx_ =
                  grad_Nx[q_point];
                const Number JxW = fe_values.JxW(q_point);

                for (const auto i : fe_values.dof_indices())
                  {
                    const unsigned int component_i =
                      fe.system_to_component_index(i).first;
                    const unsigned int i_group =
                      fe.system_to_base_index(i).first.first;

                    // compute some helper quantities
                    if (i_group == fe_data.u_dof)
                      {
                        symm_grad_Nx_i_x_tangent = symm_grad_Nx_[i] * tangent;
                        grad_Nx_i_comp_i_x_stress =
                          grad_Nx_[i][component_i] * stress_ns;
                      }

                    //// Assemble matrix
                    for (const auto j : fe_values.dof_indices_ending_at(i))
                      {
                        const unsigned int component_j =
                          fe.system_to_component_index(j).first;
                        const unsigned int j_group =
                          fe.system_to_base_index(j).first.first;

                        if ((i_group == j_group) && (i_group == fe_data.u_dof))
                          {
                            // material contribution
                            cell_matrix(i, j) +=
                              symm_grad_Nx_i_x_tangent * symm_grad_Nx_[j] * JxW;
                            // geometric contribution
                            if (component_i == component_j)
                              cell_matrix(i, j) += grad_Nx_i_comp_i_x_stress *
                                                   grad_Nx_[j][component_j] *
                                                   JxW;
                          }
                        else
                          Assert((i_group <= fe_data.u_dof) &&
                                   (j_group <= fe_data.u_dof),
                                 dealii::ExcInternalError());
                      }
                  }
              }
            // Finally, we need to copy the lower half of the local matrix into
            // the upper half:
            for (const auto i : fe_values.dof_indices())
              for (const auto j : fe_values.dof_indices_starting_at(i + 1))
                cell_matrix(i, j) = cell_matrix(j, i);

            //// process Neumann faces for pressure boundary conditions in the
            /// current configuration
            if (this->boundary_descriptor.has_pressure_boundary_condition())
              {
                for (const auto &face : element->face_iterators())
                  {
                    if (!face->at_boundary())
                      continue;

                    auto boundary_id = face->boundary_id();
                    // check if boundary.first == Pressure. Only in that case
                    // something needs to be done
                    auto boundary =
                      this->boundary_descriptor.get_pressure_boundary(
                        boundary_id);

                    if (boundary.first != Common::BoundaryType::Pressure)
                      continue;

                    fe_face_values.reinit(element, face);
                    // get displacement gradients on current element
                    fe_face_values[u_fe].get_function_gradients(
                      total_relevant_solution, displacement_gradients);
                    // compute deformation gradients at all quadrature points at
                    // once
                    HELPERS::compute_deformation_gradients(
                      displacement_gradients, F_face);

                    // get quadrature points
                    const auto quadrature_points =
                      fe_face_values.get_quadrature_points();
                    // get normal vectors in reference configuration
                    const auto N = fe_face_values.get_normal_vectors();

                    // loop over face quadrature points
                    for (const auto f_q_point :
                         fe_face_values.quadrature_point_indices())
                      {
                        // compute normal in spatial configuration
                        const auto n =
                          dealii::Physics::Transformations::nansons_formula(
                            N[f_q_point], F_face[f_q_point]);

                        // pressure boundary, evaluate function only for
                        // component 0
                        const auto neumann_value =
                          n *
                          boundary.second->value(quadrature_points[f_q_point]) *
                          fe_face_values.JxW(f_q_point);

                        // matrix contribution due to linearization of pressure
                        // boundary condition based on equation 2.180 in
                        // Holzapfel and equation 8.97 in Holzapfel in which it
                        // is described how to substitute the spatial velocity
                        // gradient in eq. 2.180 by the (spatial) gradient of
                        // the shape functions NOTE: seems to be correct
                        const auto F_inv = dealii::invert(F_face[f_q_point]);

                        // start assembly loop
                        for (const auto i : fe_values.dof_indices())
                          {
                            const unsigned int i_group =
                              fe.system_to_base_index(i).first.first;

                            // matrix contribution
                            for (const auto j : fe_values.dof_indices())
                              {
                                const unsigned int j_group =
                                  fe.system_to_base_index(j).first.first;

                                if ((i_group == j_group) &&
                                    (i_group == fe_data.u_dof))
                                  {
                                    // get gradient of shape functions in
                                    // SPATIAL configuration
                                    const auto grad_du =
                                      fe_face_values[u_fe].gradient(j,
                                                                    f_q_point) *
                                      F_inv;

                                    cell_matrix(i, j) -=
                                      fe_face_values[u_fe].value(i, f_q_point) *
                                      (dealii::trace(grad_du) * I -
                                       dealii::transpose(grad_du)) *
                                      neumann_value; // p * n * JxW
                                  }
                              }
                          }
                      }
                  }
              }
            //// and we need to assemble everything into the global matrix
            fe_data.get_zero_constraints().distribute_local_to_global(
              cell_matrix, local_dof_indices, system_state.system_matrix);
          }
      }
    // sum up overlapping parts
    system_state.system_matrix.compress(dealii::VectorOperation::add);
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  Mixture_GR<dim, VectorType, MatrixType, Number>::assemble_residual(
    const VectorType &total_solution,
    VectorType       &residual)
  {
    dealii::TimerOutput::Scope timer_section(computing_timer,
                                             "Mixture - Assemble residual");

    // need locally relevant solution
    VectorType total_relevant_solution;
    fe_data.initialize_locally_relevant_vector(total_relevant_solution);
    total_relevant_solution = total_solution;
    total_relevant_solution.update_ghost_values();

    // clear residual
    residual = 0.0;

    // needed for updating material data
    // create FEValues object
    auto fe_values = fe_data.make_fe_values(
      dealii::update_values | dealii::update_gradients |
      dealii::update_JxW_values | dealii::update_quadrature_points);
    // create vector to hold the displacement and deformation gradients (on
    // mixture level)
    std::vector<dealii::Tensor<2, dim, Number>> displacement_gradients(
      fe_values.get_quadrature().size());
    std::vector<dealii::Tensor<2, dim, Number>> F(
      fe_values.get_quadrature().size());

    auto fe_face_values = fe_data.make_fe_face_values(
      dealii::update_values | dealii::update_normal_vectors |
      dealii::update_gradients | dealii::update_quadrature_points |
      dealii::update_JxW_values);

    const auto &fe = fe_data.get_fe();

    const unsigned int dofs_per_cell = fe_data.get_fe().n_dofs_per_cell();
    const unsigned int n_q_points    = fe_data.get_quadrature().size();

    dealii::Vector<Number>                       cell_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);

    // need deformation gradients on faces
    std::vector<dealii::Tensor<2, dim, Number>> displacement_gradients_face(
      fe_face_values.get_quadrature().size());
    std::vector<dealii::Tensor<2, dim, Number>> F_face(
      fe_face_values.get_quadrature().size());

    // some temporary structures used during assembly
    std::vector<std::vector<Number>> Nx(n_q_points,
                                        std::vector<Number>(dofs_per_cell));

    std::vector<std::vector<dealii::Tensor<2, dim, Number>>> grad_Nx(
      n_q_points, std::vector<dealii::Tensor<2, dim>>(dofs_per_cell));
    std::vector<std::vector<dealii::SymmetricTensor<2, dim, Number>>>
      symm_grad_Nx(n_q_points,
                   std::vector<dealii::SymmetricTensor<2, dim>>(dofs_per_cell));

    // quadrature point coordinates
    std::vector<dealii::Point<dim>> quad_point_coords(n_q_points);

    //// EVALUATE loop over elements
    for (const auto &element :
         fe_data.get_dof_handler().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            fe_values.reinit(element);
            // get local dof indices
            element->get_dof_indices(local_dof_indices);

            // reset element right-hand side
            cell_rhs = 0.0;

            // get displacement gradients on current element
            fe_values[u_fe].get_function_gradients(total_relevant_solution,
                                                   displacement_gradients);
            // compute all deformation gradients at all quadrature points at
            // once
            HELPERS::compute_deformation_gradients(displacement_gradients, F);

            // get local mixtures of current element
            auto local_mixture = local_mixtures.get_data(element);

            // get quadrature point coordinates
            quad_point_coords = fe_values.get_quadrature_points();

            // update local mixture and fill some helper quantities
            for (const auto q_point : fe_values.quadrature_point_indices())
              {
                // get current reference growth scalar of local mixture
                const auto current_mass_fraction_ratio =
                  local_mixture[q_point]->get_current_mass_fraction_ratio();
                // compute inverse of inelastic growth deformation gradient at
                // quadrature point
                const auto F_g_inv =
                  growth_strategy->EvaluateInverseGrowthDeformationGradient(
                    current_mass_fraction_ratio, quad_point_coords[q_point]);

                // get stress contributions from growth strategy
                const auto volumetric_stress_growth =
                  growth_strategy->get_volumetric_stress_contribution(
                    current_mass_fraction_ratio, F[q_point]);
                const auto volumetric_tangent_growth =
                  growth_strategy->get_volumetric_tangent_contribution(
                    current_mass_fraction_ratio, F[q_point]);

                // update constituents at quadrature point
                local_mixture[q_point]->update_values(
                  F[q_point],
                  F_g_inv,
                  volumetric_stress_growth,
                  volumetric_tangent_growth);

                for (const unsigned int k : fe_values.dof_indices())
                  {
                    const unsigned int k_group =
                      fe.system_to_base_index(k).first.first;

                    if (k_group == fe_data.u_dof)
                      {
                        grad_Nx[q_point][k] =
                          fe_values[u_fe].gradient(k, q_point);
                        // NOTE: important to include transpose(F) for proper
                        // linearization!!! symm_grad_Nx corresponds to deltaE
                        // in Holzapfel (see page 394)
                        symm_grad_Nx[q_point][k] = dealii::symmetrize(
                          transpose(F[q_point]) * grad_Nx[q_point][k]);
                      }
                    else
                      Assert(k_group <= fe_data.u_dof,
                             dealii::ExcInternalError());
                  }
              }

            // start assembly loop
            for (const auto q_point : fe_values.quadrature_point_indices())
              {
                // get stress and tangent matrices at current quadrature point
                // from local mixture, note that these are already the
                // mass-averaged quantities.
                const dealii::SymmetricTensor<2, dim> stress =
                  local_mixture[q_point]->get_mixture_stress();

                // some abbreviations (based on step 44)
                const std::vector<dealii::SymmetricTensor<2, dim>>
                            &symm_grad_Nx_ = symm_grad_Nx[q_point];
                const Number JxW           = fe_values.JxW(q_point);

                for (const auto i : fe_values.dof_indices())
                  {
                    const unsigned int i_group =
                      fe.system_to_base_index(i).first.first;

                    //// Assemble right-hand side
                    if (i_group == fe_data.u_dof)
                      cell_rhs(i) -= (symm_grad_Nx_[i] * stress) * JxW;
                    else
                      Assert(i_group <= fe_data.u_dof,
                             dealii::ExcInternalError());
                  }
              }

            //// process Neumann faces
            for (const auto &face : element->face_iterators())
              {
                if (!face->at_boundary())
                  continue;
                // get boundary id
                auto boundary_id = face->boundary_id();

                // get boundary id and boundary function
                const auto [boundary_type, boundary_function] =
                  this->boundary_descriptor.get_boundary(boundary_id);

                fe_face_values.reinit(element, face);
                // get displacement gradients on current element
                fe_face_values[u_fe].get_function_gradients(
                  total_relevant_solution, displacement_gradients_face);
                // compute deformation gradients at all quadrature points at
                // once
                HELPERS::compute_deformation_gradients(
                  displacement_gradients_face, F_face);

                // get quadrature points
                const auto quadrature_points =
                  fe_face_values.get_quadrature_points();

                // traction vector on boundary
                dealii::Tensor<1, dim, Number> traction;

                // loop over quadrature points
                for (const auto f_q_point :
                     fe_face_values.quadrature_point_indices())
                  {
                    // Neumann boundary
                    if (boundary_type == Common::BoundaryType::Neumann)
                      {
                        // boundary_function->value(q, d) returns a scalar for
                        // each direction
                        for (unsigned int d = 0; d < dim; d++)
                          {
                            traction[d] = boundary_function->value(
                                            quadrature_points[f_q_point], d) *
                                          fe_face_values.JxW(f_q_point);
                          }
                      }
                    // pressure boundary
                    else if (boundary_type == Common::BoundaryType::Pressure)
                      {
                        // get normal vector in reference configuration
                        const auto N = fe_face_values.normal_vector(f_q_point);
                        // compute normal in spatial configuration
                        const auto n =
                          dealii::Physics::Transformations::nansons_formula(
                            N, F_face[f_q_point]);

                        // pressure boundary, evaluate function only for
                        // component 0 by not passing a second argument
                        traction = n *
                                   boundary_function->value(
                                     quadrature_points[f_q_point]) *
                                   fe_face_values.JxW(f_q_point);
                      }
                    else
                      {
                        traction *= 0.0;
                      }

                    // start assembly loop
                    for (const auto i : fe_values.dof_indices())
                      {
                        const unsigned int component_i =
                          fe.system_to_component_index(i).first;
                        const unsigned int i_group =
                          fe.system_to_base_index(i).first.first;

                        if (i_group == fe_data.u_dof)
                          cell_rhs(i) +=
                            fe_face_values.shape_value(i, f_q_point) *
                            traction[component_i];
                      }
                  }
              }

            //// and we need to assemble everything into the global residual
            /// vector
            fe_data.get_zero_constraints().distribute_local_to_global(
              cell_rhs, local_dof_indices, residual);
          }
      }

    // sum up overlapping parts
    residual.compress(dealii::VectorOperation::add);
    // I think the way I am assembling the residual is the wrong way for
    // NOX/KINSOL. NOTE: order is important! Need to compress first, then
    // multiply.
    residual *= -1.0;
  }

  // update material data on all elements
  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  Mixture_GR<dim, VectorType, MatrixType, Number>::update_material_data()
  {
    dealii::TimerOutput::Scope timer_section(
      computing_timer, "Mixture - Update quadrature point data");
    pcout << " UQPH " << std::flush;

    // update material data based on TOTAL solution
    const auto total_solution(get_total_relevant_solution());

    // create FEValues object
    auto fe_values =
      fe_data.make_fe_values(dealii::update_values | dealii::update_gradients |
                             dealii::update_quadrature_points);
    // create vector to hold the displacement and deformation gradients (on
    // mixture level)
    std::vector<dealii::Tensor<2, dim, Number>> displacement_gradients(
      fe_values.get_quadrature().size());
    std::vector<dealii::Tensor<2, dim, Number>> F(
      fe_values.get_quadrature().size());

    // quadrature point coordinates
    const auto n_q_points = fe_values.get_quadrature().size();
    std::vector<dealii::Point<dim>> quad_point_coords(n_q_points);

    // now get displacement gradients on each element and update
    for (const auto &element :
         fe_data.get_dof_handler().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            // reinit fe_values
            fe_values.reinit(element);
            // get displacement gradients on current element
            fe_values[u_fe].get_function_gradients(total_solution,
                                                   displacement_gradients);
            // compute all deformation gradients at all quadrature points at
            // once
            HELPERS::compute_deformation_gradients(displacement_gradients, F);

            // get local mixtures of current element
            auto local_mixture = local_mixtures.get_data(element);

            // get quad point coordinates
            quad_point_coords = fe_values.get_quadrature_points();

            // loop over local mixtures (same as looping over quadrature points)
            for (size_t q_point = 0; q_point < local_mixture.size(); ++q_point)
              {
                // get current reference growth scalar of local mixture
                const auto current_mass_fraction_ratio =
                  local_mixture[q_point]->get_current_mass_fraction_ratio();
                // compute inverse of inelastic growth deformation gradient at
                // quadrature point
                const auto F_g_inv =
                  growth_strategy->EvaluateInverseGrowthDeformationGradient(
                    current_mass_fraction_ratio, quad_point_coords[q_point]);

                // get stress contributions from growth strategy
                const auto volumetric_stress_growth =
                  growth_strategy->get_volumetric_stress_contribution(
                    current_mass_fraction_ratio, F[q_point]);
                const auto volumetric_tangent_growth =
                  growth_strategy->get_volumetric_tangent_contribution(
                    current_mass_fraction_ratio, F[q_point]);

                // update constituents at quadrature point
                local_mixture[q_point]->update_values(
                  F[q_point],
                  F_g_inv,
                  volumetric_stress_growth,
                  volumetric_tangent_growth);
              }
          }
      }
  }

  // update material data on all elements
  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  Mixture_GR<dim, VectorType, MatrixType, Number>::update_material_data(
    const VectorType &total_solution)
  {
    dealii::TimerOutput::Scope timer_section(
      computing_timer, "Mixture - Update quadrature point data");

    // need locally relevant solution
    VectorType total_relevant_solution;
    fe_data.initialize_locally_relevant_vector(total_relevant_solution);
    total_relevant_solution = total_solution;
    total_relevant_solution.update_ghost_values();

    // create FEValues object
    auto fe_values =
      fe_data.make_fe_values(dealii::update_values | dealii::update_gradients |
                             dealii::update_quadrature_points);
    // create vector to hold the displacement and deformation gradients (on
    // mixture level)
    std::vector<dealii::Tensor<2, dim, Number>> displacement_gradients(
      fe_values.get_quadrature().size());
    std::vector<dealii::Tensor<2, dim, Number>> F(
      fe_values.get_quadrature().size());

    // quadrature point coordinates
    const auto n_q_points = fe_values.get_quadrature().size();
    std::vector<dealii::Point<dim>> quad_point_coords(n_q_points);

    // now get displacement gradients on each element and update
    for (const auto &element :
         fe_data.get_dof_handler().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            // reinit fe_values
            fe_values.reinit(element);
            // get displacement gradients on current element
            fe_values[u_fe].get_function_gradients(total_relevant_solution,
                                                   displacement_gradients);
            // compute all deformation gradients at all quadrature points at
            // once
            HELPERS::compute_deformation_gradients(displacement_gradients, F);

            // get local mixtures of current element
            auto local_mixture = local_mixtures.get_data(element);

            // get quad point coordinates
            quad_point_coords = fe_values.get_quadrature_points();

            // loop over local mixtures (same as looping over quadrature points)
            for (size_t q_point = 0; q_point < local_mixture.size(); ++q_point)
              {
                // get current reference growth scalar of local mixture
                const auto current_mass_fraction_ratio =
                  local_mixture[q_point]->get_current_mass_fraction_ratio();
                // compute inverse of inelastic growth deformation gradient at
                // quadrature point
                const auto F_g_inv =
                  growth_strategy->EvaluateInverseGrowthDeformationGradient(
                    current_mass_fraction_ratio, quad_point_coords[q_point]);

                // get stress contributions from growth strategy
                const auto volumetric_stress_growth =
                  growth_strategy->get_volumetric_stress_contribution(
                    current_mass_fraction_ratio, F[q_point]);
                const auto volumetric_tangent_growth =
                  growth_strategy->get_volumetric_tangent_contribution(
                    current_mass_fraction_ratio, F[q_point]);

                // update constituents at quadrature point
                local_mixture[q_point]->update_values(
                  F[q_point],
                  F_g_inv,
                  volumetric_stress_growth,
                  volumetric_tangent_growth);
              }
          }
      }
  }

  // update material data of selected constituents
  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  Mixture_GR<dim, VectorType, MatrixType, Number>::update_material_data(
    const std::vector<unsigned int> &constituent_ids)
  {
    dealii::TimerOutput::Scope timer_section(
      computing_timer, "Mixture - Update quadrature point data");
    pcout << " UQPH " << std::flush;

    // update material data based on TOTAL solution
    const auto total_solution(get_total_relevant_solution());
    // create FEValues object
    auto fe_values = fe_data.make_fe_values(dealii::update_gradients |
                                            dealii::update_quadrature_points);
    // create vector to hold the displacement and deformation gradients (on
    // mixture level)
    std::vector<dealii::Tensor<2, dim, Number>> displacement_gradients(
      fe_values.get_quadrature().size());
    std::vector<dealii::Tensor<2, dim, Number>> F(
      fe_values.get_quadrature().size());

    // quadrature point coordinates
    const auto n_q_points = fe_values.get_quadrature().size();
    std::vector<dealii::Point<dim>> quad_point_coords(n_q_points);

    // now get displacement gradients on each element and update
    for (const auto &element :
         fe_data.get_dof_handler().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            // reinit fe_values
            fe_values.reinit(element);
            // get displacement gradients on current element
            fe_values[u_fe].get_function_gradients(total_solution,
                                                   displacement_gradients);
            // compute all deformation gradients at all quadrature points at
            // once
            HELPERS::compute_deformation_gradients(displacement_gradients, F);

            // get local mixtures of current element
            auto local_mixture = local_mixtures.get_data(element);

            // get quad point coordinates
            quad_point_coords = fe_values.get_quadrature_points();

            // loop over local mixtures (same as looping over quadrature points)
            for (size_t q_point = 0; q_point < local_mixture.size(); ++q_point)
              {
                // get current reference growth scalar of local mixture
                const auto current_mass_fraction_ratio =
                  local_mixture[q_point]->get_current_mass_fraction_ratio();
                // compute inverse of inelastic growth deformation gradient at
                // quadrature point
                const auto F_g_inv =
                  growth_strategy->EvaluateInverseGrowthDeformationGradient(
                    current_mass_fraction_ratio, quad_point_coords[q_point]);

                // get stress contributions from growth strategy
                const auto volumetric_stress_growth =
                  growth_strategy->get_volumetric_stress_contribution(
                    current_mass_fraction_ratio, F[q_point]);
                const auto volumetric_tangent_growth =
                  growth_strategy->get_volumetric_tangent_contribution(
                    current_mass_fraction_ratio, F[q_point]);

                // update constituents at quadrature point
                local_mixture[q_point]->update_values(
                  constituent_ids,
                  F[q_point],
                  F_g_inv,
                  volumetric_stress_growth,
                  volumetric_tangent_growth);
              }
          }
      }
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  Mixture_GR<dim, VectorType, MatrixType, Number>::update_GR()
  {
    dealii::TimerOutput::Scope timer_section(computing_timer,
                                             "Mixture - Update G&R");

    auto fe_values = fe_data.make_fe_values(dealii::update_quadrature_points);

    // quadrature point coordinates
    const auto n_q_points = fe_values.get_quadrature().size();
    std::vector<dealii::Point<dim>> quad_point_coords(n_q_points);

    // loop over all elements and update G&R
    for (const auto &element :
         fe_data.get_dof_handler().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            fe_values.reinit(element);

            // get quad point coordinates
            quad_point_coords = fe_values.get_quadrature_points();

            // get local mixtures of current element
            auto local_mixture = local_mixtures.get_data(element);

            // get local mixtures of current element and loop over it (same as
            // looping over quadrature points)


            for (size_t q_point = 0; q_point < n_q_points; ++q_point)
              {
                // get current reference growth scalar of local mixture
                const auto current_mass_fraction_ratio =
                  local_mixture[q_point]->get_current_mass_fraction_ratio();
                // compute inverse of inelastic growth deformation gradient at
                // quadrature point
                const auto F_g_inv =
                  growth_strategy->EvaluateInverseGrowthDeformationGradient(
                    current_mass_fraction_ratio, quad_point_coords[q_point]);

                // update constituents at quadrature point
                // NOTE: don't need to compute deformation gradient F again
                // since it is stored in the local_mixture. If the G&R model is
                // coupled to a diffusion problem, the local mixture needs to
                // know
                local_mixture[q_point]->update_GR(F_g_inv, coupled);
              }
          }
      }
  }

  // update transferable parameters on an element based on diffusion_values and
  // local cell collection NOTE: assumes that diffusion problem uses same
  // quadrature rule as mixture!!!
  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  Mixture_GR<dim, VectorType, MatrixType, Number>::
    update_transferable_parameters(
      const std::vector<dealii::Point<dim>>     &points,
      const Number                               time,
      const std::vector<dealii::Vector<Number>> &diffusion_values,
      const std::vector<dealii::Vector<Number>> &diffusion_laplacians,
      const std::vector<
        std::shared_ptr<Pathways::Cells::LocalCellCollection<dim, Number>>>
        &local_cell_collection,
      std::vector<std::shared_ptr<LocalMixture<dim, Number>>> &local_mixture)
  {
    dealii::TimerOutput::Scope timer_section(
      computing_timer, "Mixture - Update transferable parameters");

    // check if transformers are empty, if yes, there's nothing to do
    if (!transformers.empty())
      {
        Assert(diffusion_values.size() == local_mixture.size(),
               dealii::ExcMessage(
                 "Currently, the same quadrature rule has to be used for the "
                 "diffusion problem and mixture problem!"));

        // get number of local cell collections on the element
        const unsigned int n_cell_collections_per_element =
          local_cell_collection.size();
        Assert(n_cell_collections_per_element == 1 or
                 n_cell_collections_per_element == local_mixture.size(),
               dealii::ExcMessage(
                 "Currently only 1 or n_quad_points cells are supported!"));

        // loop over local mixtures (same as looping over quadrature points)
        for (size_t q_point = 0; q_point < local_mixture.size(); ++q_point)
          {
            // get state of diffusion problem at quadrature point
            const auto &diffusion_state = diffusion_values[q_point];
            const auto &diffusion_laplacian_state =
              diffusion_laplacians[q_point];

            // get local cell collection
            const auto &loc_cell_collection =
              n_cell_collections_per_element == 1 ?
                *local_cell_collection[0] :
                *local_cell_collection[q_point];

            // loop over the constituents in the local mixture of the current
            // quadrature point
            for (const auto &[constituent_id, constituent] :
                 local_mixture[q_point]->get_constituents())
              {
                // get constituent id
                // const auto constituent_id =
                // constituent->get_constituent_id();
                // check if constituent has transformers at all (hyperelastic
                // constituents for example probably do not have transformers,
                // so we can skip them)
                if (transformers.find(constituent_id) != transformers.end())
                  {
                    // get transferable parameters of that constituent
                    auto &transferable_parameters =
                      constituent->get_transferable_parameters();
                    // reset mass_production rate of current constituent
                    // needed because we might use "+=" in the transformer of a
                    // constituent that is produced by several cells (e.g. FBs
                    // and SMCs might both be present at a quadrature point and
                    // can both produce collagen)
                    transferable_parameters.mass_production = 0.0;

                    // set trace of sigma_h in transferable parameters
                    const auto mixture_PK2_stress =
                      local_mixture[q_point]->get_mixture_stress();
                    const auto mixture_F =
                      local_mixture[q_point]
                        ->get_mixture_deformation_gradient();
                    // compute sigma and its trace
                    const auto cauchy_stress_mixture =
                      1. / dealii::determinant(mixture_F) * mixture_F *
                      mixture_PK2_stress * dealii::transpose(mixture_F);
                    const auto trace_sigma =
                      dealii::trace(cauchy_stress_mixture);
                    // set
                    transferable_parameters.trace_sigma_h = trace_sigma;

                    // get transformers (one for each cell type) associated with
                    // that constituent
                    for (const auto &[cell_type, transformer] :
                         transformers[constituent_id])
                      {
                        // if cell type is present in the local cell collection,
                        // get its pathway output and baseline pathway output
                        // and pass it to the transformer todo: is the "if"
                        // necessary? Could it be the case that a cell type
                        //  is not present? I think it is necessary since we
                        //  could have SMCs and FBs producing collagen but one
                        //  of them might not be present in the current local
                        //  cell collection
                        if (loc_cell_collection.cell_type_exists(cell_type))
                          {
                            // now we can apply the transformer
                            transformer(
                              points[q_point],
                              time,
                              diffusion_state,
                              diffusion_laplacian_state,
                              loc_cell_collection.get_average_pathway_output(
                                cell_type),
                              loc_cell_collection
                                .get_average_pathway_baseline_output(cell_type),
                              transferable_parameters);
                          }
                      }
                  }
              }
          }
      }
  }

  // update prestretched of iteratively prestressed constituents on all elements
  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  Mixture_GR<dim, VectorType, MatrixType, Number>::update_prestretch(
    const std::vector<unsigned int> &constituent_ids)
  {
    // get total solution
    const auto total_solution(get_total_relevant_solution());

    // create FEValues object
    auto fe_values = fe_data.make_fe_values(dealii::update_gradients |
                                            dealii::update_quadrature_points);
    // create vector to hold the displacement and deformation gradients (on
    // mixture level)
    std::vector<dealii::Tensor<2, dim, Number>> displacement_gradients(
      fe_values.get_quadrature().size());
    std::vector<dealii::Tensor<2, dim, Number>> F(
      fe_values.get_quadrature().size());

    // quadrature point coordinates
    const auto n_q_points = fe_values.get_quadrature().size();
    std::vector<dealii::Point<dim>> quad_point_coords(n_q_points);

    // now get displacement gradients on each element and update
    for (const auto &element :
         fe_data.get_dof_handler().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            // reinit fe_values
            fe_values.reinit(element);
            // get displacement gradients on current element
            fe_values[u_fe].get_function_gradients(total_solution,
                                                   displacement_gradients);
            // compute all deformation gradients at all quadrature points at
            // once
            HELPERS::compute_deformation_gradients(displacement_gradients, F);

            // get local mixtures of current element
            auto local_mixture = local_mixtures.get_data(element);

            // get quadrature point coordinates
            quad_point_coords = fe_values.get_quadrature_points();

            // loop over local mixtures (same as looping over quadrature points)
            for (size_t q_point = 0; q_point < local_mixture.size(); ++q_point)
              {
                // update constituents at quadrature point
                local_mixture[q_point]->update_prestretch(
                  constituent_ids,
                  F[q_point],
                  quad_point_coords[q_point],
                  *prestretch_strategy);
              }
          }
      }
  }

  // update prestretch of iteratively prestressed constituents on all elements
  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  Mixture_GR<dim, VectorType, MatrixType, Number>::evaluate_and_set_prestretch()
  {
    // create FEValues object with update quadrature points
    auto fe_values = fe_data.make_fe_values(dealii::update_quadrature_points);

    // loop over elements and evaluate new prestretch
    for (const auto &element :
         fe_data.get_dof_handler().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            // reinit fe_values
            fe_values.reinit(element);
            // get coordinates of quadrature points
            const auto &quadrature_points = fe_values.get_quadrature_points();

            // get local mixtures of current element
            auto local_mixture = local_mixtures.get_data(element);

            // loop over local mixtures (same as looping over quadrature points)
            for (size_t q_point = 0; q_point < local_mixture.size(); ++q_point)
              {
                // update constituents at quadrature point
                local_mixture[q_point]->evaluate_prestretch(
                  quadrature_points[q_point]);
              }
          }
      }
  }

  // Debug only
  // print constituent names at quadrature points
  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  Mixture_GR<dim, VectorType, MatrixType, Number>::print_qp_data()
  {
    pcout << "Checking quadrature point data..." << std::endl;
    // setup quadrature point data of constituents
    const auto n_q_points = fe_data.get_quadrature().size();

    for (const auto &element :
         fe_data.get_dof_handler().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            const std::vector<std::shared_ptr<LocalMixture<dim, Number>>>
              qp_data = local_mixtures.get_data(element);
            Assert(qp_data.size() == n_q_points, dealii::ExcInternalError());

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              qp_data[q_point]->print_constituents();
          }
      }
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  VectorType
  Mixture_GR<dim, VectorType, MatrixType, Number>::get_total_relevant_solution()
    const
  {
    VectorType solution_total = system_state.get_total_solution();

    // copy to locally relevant solution and return that
    VectorType total_relevant_solution(fe_data.get_locally_owned_dofs(),
                                       fe_data.get_locally_relevant_dofs(),
                                       mpi_communicator);

    total_relevant_solution = solution_total;
    total_relevant_solution.update_ghost_values();

    return total_relevant_solution;
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  Postprocessing::Postprocessor<dim, Number> &
  Mixture_GR<dim, VectorType, MatrixType, Number>::get_postprocessor()
  {
    return mixture_postprocessor;
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  Common::BoundaryDescriptor<dim, Number> &
  Mixture_GR<dim, VectorType, MatrixType, Number>::get_new_boundary_descriptor()
  {
    return boundary_descriptor;
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  Mixture_GR<dim, VectorType, MatrixType, Number>::output_results(
    const unsigned int step)
  {
    dealii::TimerOutput::Scope timer_section(computing_timer,
                                             "Mixture - Output results");

    // create locally relevant solution
    VectorType total_relevant_solution(get_total_relevant_solution());

    // delegate output to postprocessor
    mixture_postprocessor.write_output(step,
                                       fe_data.get_dof_handler(),
                                       total_relevant_solution,
                                       local_mixtures);
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  bool
  Mixture_GR<dim, VectorType, MatrixType, Number>::is_coupled() const
  {
    return coupled;
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  Mixture_GR<dim, VectorType, MatrixType, Number>::print_timer_stats(
    const bool print_mpi_stats) const
  {
    pcout << "Mixture Timings";
    // always print summary
    computing_timer.print_summary();
    // optionally, print mpi stats
    if (print_mpi_stats)
      computing_timer.print_wall_time_statistics(mpi_communicator);
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  Mixture_GR<dim, VectorType, MatrixType, Number>::
    write_prestretched_configuration(const std::string &output_filename,
                                     const unsigned int constituent_id) const
  {
    pcout << std::endl << "Writing prestretched configuration..." << std::endl;

    // create solution tansfer object
    dealii::parallel::distributed::SolutionTransfer<dim, VectorType> sol_trans(
      fe_data.get_dof_handler());

    // solution transfer needs solution with ghost values for serialization
    const auto total_solution(get_total_relevant_solution());
    sol_trans.prepare_for_serialization(total_solution);

    // write triangulation and solution vector to file
    // cast to p::d::T
    const auto &tria =
      dynamic_cast<const dealii::parallel::distributed::Triangulation<dim> &>(
        fe_data.get_triangulation());

    //// write quadrature point data
    const unsigned int n_components_per_tensor = dim * dim;
    const unsigned int n_q_points = fe_data.get_quadrature().size();
    // create vectors to store data
    std::vector<dealii::Vector<Number>> temp_data(n_q_points *
                                                  n_components_per_tensor);
    // initialize vectors
    for (auto &v : temp_data)
      v.reinit(fe_data.get_triangulation().n_active_cells());

    for (const auto &element :
         fe_data.get_triangulation().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            // get LocalMixture data
            const auto qp_data = local_mixtures.get_data(element);

            // loop over quadrature points
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              {
                // get prestretch_tensor of constituent with given
                // constituent_id
                const auto &prestretch_tensor =
                  qp_data[q_point]
                    ->get_constituent(constituent_id)
                    ->get_prestretch_tensor();
                // loop over components of tensor
                for (unsigned int i = 0; i < n_components_per_tensor; ++i)
                  temp_data[i + n_components_per_tensor *
                                  q_point][element->active_cell_index()] =
                    prestretch_tensor[dealii::Tensor<2, dim>::
                                        unrolled_to_component_indices(i)];
              }
          }
      }

    // need to copy data to this format
    std::vector<const dealii::Vector<Number> *> data_to_transfer(
      n_q_points * n_components_per_tensor);
    // copy data from temp_data
    unsigned int i = 0;
    for (auto &v : data_to_transfer)
      {
        v = &temp_data[i];
        ++i;
      }

    // prepare data for serialization
    dealii::parallel::distributed::
      CellDataTransfer<dim, dim, dealii::Vector<Number>>
        cell_data_trans(tria);
    cell_data_trans.prepare_for_serialization(data_to_transfer);

    // save
    tria.save(output_filename);

    pcout << "Prestretched configuration written!" << std::endl;
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  Mixture_GR<dim, VectorType, MatrixType, Number>::
    read_prestretched_configuration(const unsigned int constituent_id)
  {
    pcout << std::endl << "Reading prestretched configuration..." << std::endl;

    // NOTE: important that the dof_handler used here is associated with the
    // triangulation!
    dealii::parallel::distributed::SolutionTransfer<dim, VectorType> sol_trans(
      fe_data.get_dof_handler());
    // deserialize - IMPORTANT: solution must not contain ghost values!
    sol_trans.deserialize(system_state.solution);


    //// read cell-wise data
    // prepare triangulation
    // cast to p::d::T
    const auto &tria =
      dynamic_cast<const dealii::parallel::distributed::Triangulation<dim> &>(
        fe_data.get_triangulation());
    const unsigned int n_components_per_tensor = dim * dim;
    const unsigned int n_q_points = fe_data.get_quadrature().size();
    // create vectors for storing temporary data
    std::vector<dealii::Vector<Number>> transferred_data_temp(
      n_q_points * n_components_per_tensor);
    // initialize vectors to correct size
    for (auto &v : transferred_data_temp)
      v.reinit(tria.n_active_cells());
    // need this specific format to read data
    std::vector<dealii::Vector<Number> *> transferred_data(
      n_q_points * n_components_per_tensor);

    unsigned int j = 0;
    for (auto &v : transferred_data)
      {
        v = &transferred_data_temp[j];
        ++j;
      }

    // create CellDataTransfer object
    dealii::parallel::distributed::
      CellDataTransfer<dim, dim, dealii::Vector<Number>>
        cell_data_trans(tria);
    // deserialize data
    cell_data_trans.deserialize(transferred_data);

    // temp vector to hold all the data of the tensor
    dealii::Vector<Number> temp_vector(n_components_per_tensor);

    for (const auto &element : tria.active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            // get quad point data
            auto qp_data = local_mixtures.get_data(element);

            // loop over quadrature points
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              {
                // reset temp_vector
                temp_vector = 0.0;
                // loop over components of the tensor
                for (unsigned int i = 0; i < n_components_per_tensor; ++i)
                  {
                    temp_vector[i] =
                      transferred_data_temp[i +
                                            n_components_per_tensor * q_point]
                                           [element->active_cell_index()];
                  }
                // set tensor in new quad point data
                qp_data[q_point]
                  ->get_constituent(constituent_id)
                  ->get_prestretch_tensor() =
                  dealii::Tensor<2, dim>(make_array_view(temp_vector));
              }
          }
      }

    // need to call update material data to update all the constituents based on
    // the final solution found during Prestretching. Has to be done after
    // setting the prestretch values of elastin.
    update_material_data();

    pcout << std::endl << "Prestretched configuration read!" << std::endl;
  }

  // explicit instantiations
  template class Mixture_GR<3,
                            dealii::TrilinosWrappers::MPI::Vector,
                            dealii::TrilinosWrappers::SparseMatrix,
                            double>;

} // namespace Mixture