#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <MSHCMM/common/boundary_descriptor.h>
#include <MSHCMM/diffusion/diffusion_manager.h>
#include <MSHCMM/diffusion/diffusion_parameters.h>
#include <MSHCMM/diffusion/diffusion_parameters_constant.h>
#include <MSHCMM/diffusion/diffusion_parameters_pathway.h>
#include <MSHCMM/mixture/constituents/cylindrical_coordinate_transformer.h>
#include <MSHCMM/mixture/growth_and_remodeling/local_mixture.h>
#include <MSHCMM/pathways/cells/local_cell_collection.h>
#include <MSHCMM/pathways/endothelial_cell_layer.h>
#include <MSHCMM/pathways/pathway_manager.h>

#include <string>

#include "arkode/arkode_arkstep.h"

namespace Diffusion
{

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  DiffusionManager<dim, VectorType, MatrixType, Number>::DiffusionManager(
    const dealii::parallel::TriangulationBase<dim> &triangulation,
    const unsigned int                              fe_degree,
    const unsigned int                              quad_degree,
    const unsigned int                              n_components,
    const std::shared_ptr<dealii::Function<dim>>   &initial_condition,
    const std::shared_ptr<dealii::Function<dim>>   &source_term,
    const Common::BoundaryDescriptor<dim>          &boundary_descriptor)
    : mpi_communicator(triangulation.get_communicator())
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(pcout,
                      dealii::TimerOutput::never,
                      dealii::TimerOutput::wall_times)
    , fe_data(triangulation, fe_degree, quad_degree, n_components)
    , diffusion_parameters(
        std::make_unique<DiffusionParametersPathway<dim, Number>>(
          triangulation,
          fe_data.get_quadrature().size(),
          n_components))
    , initial_condition(initial_condition)
    , source_term(source_term)
    , boundary_descriptor(boundary_descriptor)
  {
    // initialize component names for output
    for (unsigned int i = 0; i < this->fe_data.get_fe().n_components(); ++i)
      output_component_names.push_back("component_" + std::to_string(i));
  }


  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  DiffusionManager<dim, VectorType, MatrixType, Number>::setup_system(
    const std::shared_ptr<dealii::Function<dim>> &diffusion_coefficients,
    const std::vector<std::vector<Number>>       &linear_coefficients,
    const HigherOrderMatrix<Number>              &higher_order_coefficients,
    const HigherOrderMatrix<Number> &higher_order_coefficients_derivative,
    const bool                       enable_pathway_dependence)
  {
    // print some statistics
    pcout << "Number of global active elements in DiffusionManager: "
          << fe_data.get_triangulation().n_global_active_cells() << std::endl;
    pcout << "Number of global dofs in DiffusionManager: "
          << fe_data.get_dof_handler().n_dofs() << std::endl;

    // initialize system vectors and matrices
    fe_data.get_constraints().clear();
    // build dirichlet constraints. Note: constraints.close() is called in that
    // function no need to change Newton iteration
    boundary_descriptor.build_dirichlet_constraints(fe_data.get_dof_handler(),
                                                    fe_data.get_constraints(),
                                                    false /*homogeneous*/);

    // initialize matrices
    fe_data.initialize_matrix(system_matrix);
    fe_data.initialize_matrix(laplace_matrix);
    fe_data.initialize_matrix(K);
    fe_data.initialize_matrix(jacobian);
    fe_data.initialize_matrix(mass_matrix);

    // initialize vectors (locally owned)
    fe_data.initialize_locally_owned_vector(solution);
    fe_data.initialize_locally_owned_vector(system_rhs);

    // interpolate initial condition
    interpolate_initial_conditions();

    // assemble matrices
    assemble_mass_matrix();
    assemble_laplace_matrix(diffusion_coefficients.get());

    // create LocalDiffusionParameters
    if (enable_pathway_dependence)
      diffusion_parameters =
        std::make_unique<DiffusionParametersPathway<dim, Number>>(
          fe_data.get_triangulation(),
          fe_data.get_quadrature().size(),
          fe_data.get_fe().n_components(),
          linear_coefficients,
          higher_order_coefficients,
          higher_order_coefficients_derivative);
    else
      {
        // create (constant) diffusion parameters
        diffusion_parameters =
          std::make_unique<DiffusionParametersConstant<dim, Number>>(
            linear_coefficients,
            higher_order_coefficients,
            higher_order_coefficients_derivative);
        diffusion_parameters->set_size(fe_data.get_quadrature().size());
      }
  }



  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  DiffusionManager<dim, VectorType, MatrixType, Number>::assemble_mass_matrix()
  {
    MatrixCreator::create_mass_matrix(
      fe_data.get_mapping(),
      fe_data.get_dof_handler(),
      fe_data.get_quadrature(),
      mass_matrix,
      (const Function<dim, Number> *const)nullptr,
      fe_data.get_constraints());
    // necessary?
    mass_matrix.compress(VectorOperation::add);
  }



  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  DiffusionManager<dim, VectorType, MatrixType, Number>::
    assemble_laplace_matrix(const Function<dim, Number> *const d)
  {
    // reset
    laplace_matrix = 0.0;
    // do interpolation
    if (d)
      MatrixCreator::create_laplace_matrix(fe_data.get_mapping(),
                                           fe_data.get_dof_handler(),
                                           fe_data.get_quadrature(),
                                           laplace_matrix,
                                           d,
                                           fe_data.get_constraints());
    else
      MatrixCreator::create_laplace_matrix(
        fe_data.get_mapping(),
        fe_data.get_dof_handler(),
        fe_data.get_quadrature(),
        laplace_matrix,
        (const Function<dim, Number> *const)nullptr,
        fe_data.get_constraints());
    // necessary?
    laplace_matrix.compress(VectorOperation::add);
    // change sign
    laplace_matrix *= -1.0;
  }


  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  DiffusionManager<dim, VectorType, MatrixType, Number>::
    assemble_laplace_matrix(
      const std::vector<std::shared_ptr<TensorFunction<2, dim, Number>>> &d,
      const Mixture::Constituents::CylindricalCoordinateTransformer<dim, Number>
        &cos_transformer)
  {
    TimerOutput::Scope t(computing_timer,
                         "DiffusionManager - Assemble laplace matrix");

    // create FEValues
    auto fe_values = fe_data.make_fe_values(dealii::update_gradients |
                                            dealii::update_quadrature_points |
                                            dealii::update_JxW_values);
    // reset matrix
    laplace_matrix = 0.;

    const unsigned int dofs_per_cell = fe_data.get_fe().n_dofs_per_cell();
    const unsigned int n_q_points    = fe_data.get_quadrature().size();
    dealii::FullMatrix<Number> element_laplace_matrix(dofs_per_cell,
                                                      dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);

    // allocate memory for quadrature points
    std::vector<Point<dim>> quadrature_points(n_q_points);

    for (const auto &element :
         fe_data.get_dof_handler().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            element_laplace_matrix = 0.;

            fe_values.reinit(element);

            // get quadrature points
            quadrature_points = fe_values.get_quadrature_points();

            for (const auto q_point : fe_values.quadrature_point_indices())
              {
                for (const auto i : fe_values.dof_indices())
                  {
                    // get component index of i
                    const auto comp_i =
                      fe_data.get_fe().system_to_component_index(i).first;

                    // get diffusion coefficient of current component
                    const auto D = d[comp_i]->value(quadrature_points[q_point]);
                    const auto D_rot =
                      cos_transformer.to_cartesian(quadrature_points[q_point],
                                                   D);

                    for (const auto j : fe_values.dof_indices())
                      {
                        // get component index of j
                        const auto comp_j =
                          fe_data.get_fe().system_to_component_index(j).first;

                        if (comp_i == comp_j)
                          element_laplace_matrix(i, j) -=
                            D_rot * fe_values.shape_grad(i, q_point) // phi_i
                            * fe_values.shape_grad(j, q_point)       // phi_j
                            * fe_values.JxW(q_point);                // dx
                      }
                  }
              }
            // distribute local-to-global
            element->get_dof_indices(local_dof_indices);
            fe_data.get_constraints().distribute_local_to_global(
              element_laplace_matrix, local_dof_indices, laplace_matrix);
          }
      }

    // sum up overlapping parts
    laplace_matrix.compress(dealii::VectorOperation::add);
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  DiffusionManager<dim, VectorType, MatrixType, Number>::
    assemble_source_and_boundary_terms(const Number time,
                                       const bool   use_single_cell_collection)
  {
    TimerOutput::Scope t(
      computing_timer, "DiffusionManager - Assemble source and boundary terms");

    // update time in source terms and boundary descriptor
    update_source_term_time(time);
    boundary_descriptor.set_evaluation_time(time);

    // reset rhs
    system_rhs = 0.0;

    // FEValues objects
    auto fe_values = fe_data.make_fe_values(
      update_values | update_quadrature_points | update_JxW_values);
    auto fe_face_values =
      fe_data.make_fe_face_values(update_values | update_quadrature_points |
                                  update_JxW_values | update_normal_vectors);

    // some helper quantities
    const unsigned int dofs_per_cell   = fe_data.get_fe().n_dofs_per_cell();
    const unsigned int n_q_points      = fe_data.get_quadrature().size();
    const unsigned int n_q_points_face = fe_face_values.get_quadrature().size();
    // element vector and local dof indices
    Vector<Number>                       element_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // allocate memory for quadrature point coordinates depending on the usage
    // of single_cell_collection
    std::vector<Point<dim>> quadrature_points(
      use_single_cell_collection ? 1 : n_q_points);
    std::vector<Point<dim>> quadrature_points_face(
      use_single_cell_collection ? 1 : n_q_points_face);

    // endothelial cells
    std::vector<
      std::shared_ptr<Pathways::Cells::LocalCellCollection<dim, Number>>>
      local_endothelial_cell_collection(
        use_single_cell_collection ? 1 : n_q_points_face);
    // EMPTY mixture since this model only couples DiffusionManager and Pathways
    std::vector<std::shared_ptr<Mixture::LocalMixture<dim, Number>>>
      local_mixture;


    // loop over cells
    for (const auto &element :
         fe_data.get_dof_handler().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            // reset element
            element_rhs = 0.;

            // reinit fe_values to current cell and get current function values
            fe_values.reinit(element);

            // get quadrature point coordinates
            // note that if we use a single cell collection, we just create a
            // vector of size n_q_points with the coordinates of the cell center
            quadrature_points =
              use_single_cell_collection ?
                std::vector<dealii::Point<dim>>(
                  1, element->center(true /*respect_manifold*/)) :
                fe_values.get_quadrature_points();

            // assemble right-hand side if source term exists
            if (source_term)
              assemble_source_term(fe_values, quadrature_points, element_rhs);

            //// process Neumann faces
            for (const auto &face : element->face_iterators())
              {
                if (!face->at_boundary())
                  continue;

                auto boundary_id = face->boundary_id();

                // check if boundary.first == Neumann. Only in that case
                // something needs to be done
                auto boundary =
                  this->boundary_descriptor.get_neumann_boundary(boundary_id);

                if (boundary.first != Common::BoundaryType::Neumann)
                  continue;

                // reinit FEFaceValues
                fe_face_values.reinit(element, face);

                // get quadrature points
                quadrature_points_face = fe_face_values.get_quadrature_points();

                assemble_neumann_bc(fe_face_values,
                                    *(boundary.second),
                                    quadrature_points_face,
                                    element_rhs);
              }

            // distribute local-to-global
            element->get_dof_indices(local_dof_indices);
            fe_data.get_constraints().distribute_local_to_global(
              element_rhs, local_dof_indices, system_rhs);
          }
      }

    // sum up overlapping parts
    system_rhs.compress(VectorOperation::add);
  }


  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  DiffusionManager<dim, VectorType, MatrixType, Number>::assemble_source_term(
    const FEValues<dim>           &fe_values,
    const std::vector<Point<dim>> &quadrature_points,
    Vector<Number>                &element_rhs)
  {
    for (const auto q_point : fe_values.quadrature_point_indices())
      {
        for (const auto i : fe_values.dof_indices())
          {
            // get component index of i
            const auto comp_i =
              fe_values.get_fe().system_to_component_index(i).first;

            // assemble local right hand side
            element_rhs(i) +=
              fe_values.shape_value(i, q_point) *
              source_term->value(quadrature_points[q_point], comp_i) *
              fe_values.JxW(q_point);
          }
      }
  }

  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  DiffusionManager<dim, VectorType, MatrixType, Number>::assemble_neumann_bc(
    const FEFaceValues<dim>       &fe_face_values,
    const Function<dim, Number>   &function,
    const std::vector<Point<dim>> &quadrature_points,
    Vector<Number>                &element_rhs) const
  {
    const auto normals = fe_face_values.get_normal_vectors();

    for (const auto f_q_point : fe_face_values.quadrature_point_indices())
      {
        for (const auto i : fe_face_values.dof_indices())
          {
            const auto comp_i =
              fe_face_values.get_fe().system_to_component_index(i).first;

            element_rhs(i) +=
              fe_face_values.shape_value(i, f_q_point) *
              function.value(quadrature_points[f_q_point], comp_i) *
              fe_face_values.JxW(f_q_point);
          }
      }
  }


  template <int dim, typename VectorType, typename MatrixType, typename Number>
  dealii::Vector<Number>
  DiffusionManager<dim, VectorType, MatrixType, Number>::
    assemble_endothelial_contribution(
      Pathways::PathwayManager<dim, Number> &pathway_manager)
  {
    TimerOutput::Scope t(
      computing_timer, "DiffusionManager - Assemble endothelial contribution");

    auto fe_face_values =
      fe_data.make_fe_face_values(update_values | update_JxW_values);

    // some helper quantities
    const unsigned int dofs_per_cell   = fe_data.get_fe().n_dofs_per_cell();
    const unsigned int n_q_points_face = fe_face_values.get_quadrature().size();
    // element vector and local dof indices
    Vector<Number>                       element_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // allocate memory for endothelial cells
    std::vector<
      std::shared_ptr<Pathways::Cells::LocalCellCollection<dim, Number>>>
      local_endothelial_cell_collection(n_q_points_face);

    dealii::Vector<Number> boundary_values(
      fe_face_values.get_fe().n_components());

    // loop over cells
    for (const auto &element :
         fe_data.get_dof_handler().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            // reset element
            element_rhs = 0.;

            //// process endothelial cell layer
            for (const auto &face : element->face_iterators())
              {
                if (!face->at_boundary())
                  continue;

                if (face->boundary_id() ==
                    pathway_manager.get_endothelium_boundary_id())
                  {
                    // reinit fe_face_values to current cell and get current
                    // function values
                    fe_face_values.reinit(element, face);
                    // get endothelial cells on the current element
                    local_endothelial_cell_collection =
                      pathway_manager.get_local_endothelial_cell_collection(
                        element);
                    // get output transformer, from endothelial pathway output
                    // to diffusion problem
                    const auto &transformer =
                      pathway_manager.get_endothelial_cell_layer()
                        ->get_output_transformer();
                    // get endothelial cell type
                    const auto &cell_type =
                      pathway_manager.get_endothelial_cell_layer()
                        ->get_endothelial_cell_type();
                    // assemble face contribution (in essence the endothelial
                    // cells represent a Neumann boundary)
                    boundary_values = assemble_endothelial_face(
                      fe_face_values,
                      local_endothelial_cell_collection,
                      transformer,
                      cell_type,
                      element_rhs);
                  }
              }

            // distribute local-to-global
            element->get_dof_indices(local_dof_indices);
            fe_data.get_constraints().distribute_local_to_global(
              element_rhs, local_dof_indices, system_rhs);
          }
      }

    // sum up overlapping parts
    system_rhs.compress(VectorOperation::add);

    return boundary_values;
  }


  template <int dim, typename VectorType, typename MatrixType, typename Number>
  dealii::Vector<Number>
  DiffusionManager<dim, VectorType, MatrixType, Number>::
    assemble_endothelial_face(
      const dealii::FEFaceValues<dim> &fe_face_values,
      const std::vector<
        std::shared_ptr<Pathways::Cells::LocalCellCollection<dim, Number>>>
        &local_endothelial_cell_collection,
      const Pathways::OutputTransformerEndothelialPathway<dim, Number>
                                      &transformer,
      const Pathways::Cells::CellType &cell_type,
      dealii::Vector<Number>          &element_rhs) const
  {
    // create vector to be filled by output transformer
    dealii::Vector<Number> boundary_values(
      fe_face_values.get_fe().n_components());

    // loop over quadrature points
    for (const auto q_point : fe_face_values.quadrature_point_indices())
      {
        // get local cell collection at current quadrature point
        // in case a single cell collection is used, we only have one entry in
        // the vector
        const auto &loc_cell_collection =
          local_endothelial_cell_collection.size() == 1 ?
            *local_endothelial_cell_collection[0] :
            *local_endothelial_cell_collection[q_point];

        // apply transformer from endothelial pathway output to diffusion
        // problem
        transformer(loc_cell_collection.get_average_pathway_output(cell_type),
                    loc_cell_collection.get_average_pathway_baseline_output(
                      cell_type),
                    boundary_values);

        // loop over dofs
        for (const auto i : fe_face_values.dof_indices())
          {
            // get component index of dof
            const unsigned int component_i =
              fe_face_values.get_fe().system_to_component_index(i).first;

            element_rhs(i) += fe_face_values.shape_value(i, q_point) *
                              boundary_values[component_i] *
                              fe_face_values.JxW(q_point);
          }
      }

    return boundary_values;
  }



  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  DiffusionManager<dim, VectorType, MatrixType, Number>::
    assemble_outlfux_boundary(const dealii::Vector<Number> &influx_values,
                              const double                  outflux_fraction,
                              const unsigned int            outflux_boundary_id)
  {
    TimerOutput::Scope t(computing_timer,
                         "DiffusionManager - Assemble outflux boundary");

    auto fe_face_values =
      fe_data.make_fe_face_values(update_values | update_JxW_values);

    // some helper quantities
    const unsigned int dofs_per_cell = fe_data.get_fe().n_dofs_per_cell();
    // element vector and local dof indices
    Vector<Number>                       element_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // scale by outflux_fraction, need the minus to make it outflux
    dealii::Vector<Number> outflux(fe_face_values.get_fe().n_components());
    outflux.equ(-outflux_fraction, influx_values);

    // loop over cells
    for (const auto &element :
         fe_data.get_dof_handler().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            // reset element
            element_rhs = 0.;

            //// process outflux boundary
            for (const auto &face : element->face_iterators())
              {
                if (!face->at_boundary())
                  continue;

                if (face->boundary_id() == outflux_boundary_id)
                  {
                    // reinit fe_face_values to current cell and get current
                    // function values
                    fe_face_values.reinit(element, face);
                    // assemble outflux face contribution (related to the influx
                    // from the ECs)
                    assemble_outflux_face(fe_face_values, outflux, element_rhs);
                  }
              }

            // distribute local-to-global
            element->get_dof_indices(local_dof_indices);
            fe_data.get_constraints().distribute_local_to_global(
              element_rhs, local_dof_indices, system_rhs);
          }
      }

    // sum up overlapping parts
    system_rhs.compress(VectorOperation::add);
  }


  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  DiffusionManager<dim, VectorType, MatrixType, Number>::assemble_outflux_face(
    const dealii::FEFaceValues<dim> &fe_face_values,
    const dealii::Vector<Number>    &outflux,
    dealii::Vector<Number>          &element_rhs) const
  {
    // loop over quadrature points
    for (const auto q_point : fe_face_values.quadrature_point_indices())
      {
        // loop over dofs
        for (const auto i : fe_face_values.dof_indices())
          {
            // get component index of dof
            const unsigned int component_i =
              fe_face_values.get_fe().system_to_component_index(i).first;

            element_rhs(i) += fe_face_values.shape_value(i, q_point) *
                              outflux[component_i] *
                              fe_face_values.JxW(q_point);
          }
      }
  }


  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  DiffusionManager<dim, VectorType, MatrixType, Number>::assemble_K(
    const VectorType &current_solution)
  {
    TimerOutput::Scope t(computing_timer, "DiffusionManager - Assemble K");

    // create FEValues
    auto fe_values =
      fe_data.make_fe_values(dealii::update_values | dealii::update_JxW_values);
    // reset matrix
    K = 0.;

    const unsigned int dofs_per_cell = fe_data.get_fe().n_dofs_per_cell();
    const unsigned int n_q_points    = fe_data.get_quadrature().size();
    dealii::FullMatrix<Number> element_K_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);

    // allocate memory for
    // - local diffusion parameters
    std::vector<std::shared_ptr<LocalDiffusionParameters<Number>>> ldp(
      n_q_points);
    // - current solution at quadrature points
    std::vector<Vector<Number>> diffusion_values(
      n_q_points, Vector<Number>(fe_data.get_fe().components));

    // get current (locally relevant) solution
    VectorType current_relevant_solution;
    fe_data.initialize_locally_relevant_vector(current_relevant_solution);
    current_relevant_solution = current_solution;

    for (const auto &element :
         fe_data.get_dof_handler().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            element_K_matrix = 0.;

            fe_values.reinit(element);
            fe_values.get_function_values(current_relevant_solution,
                                          diffusion_values);

            // get local diffusion parameters
            ldp = diffusion_parameters->get_local_diffusion_parameters(element);

            for (const auto q_point : fe_values.quadrature_point_indices())
              {
                for (const auto i : fe_values.dof_indices())
                  {
                    // get component index of i
                    const auto comp_i =
                      fe_data.get_fe().system_to_component_index(i).first;

                    for (const auto j : fe_values.dof_indices())
                      {
                        // get component index of j
                        const auto comp_j =
                          fe_data.get_fe().system_to_component_index(j).first;

                        element_K_matrix(i, j) +=
                          ((ldp[q_point]->get_linear_coefficient(
                              comp_i, comp_j) + // linear terms
                            ldp[q_point]->get_quadratic_coefficient(
                              comp_i,
                              comp_j,
                              diffusion_values[q_point]) // higher order terms
                            ) *
                           fe_values.shape_value(i, q_point) * // phi_i
                           fe_values.shape_value(j, q_point))  // phi_j
                          * fe_values.JxW(q_point);            // dx
                      }
                  }
              }
            // distribute local-to-global
            element->get_dof_indices(local_dof_indices);
            fe_data.get_constraints().distribute_local_to_global(
              element_K_matrix, local_dof_indices, K);
          }
      }

    // sum up overlapping parts
    K.compress(dealii::VectorOperation::add);
  }


  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  DiffusionManager<dim, VectorType, MatrixType, Number>::assemble_jacobian(
    const VectorType &current_solution)
  {
    TimerOutput::Scope t(computing_timer,
                         "DiffusionManager - Assemble jacobian");

    // create FEValues
    auto fe_values =
      fe_data.make_fe_values(dealii::update_values | dealii::update_JxW_values);
    // reset matrix
    jacobian = 0.;

    const unsigned int dofs_per_cell = fe_data.get_fe().n_dofs_per_cell();
    const unsigned int n_q_points    = fe_data.get_quadrature().size();
    dealii::FullMatrix<Number> element_jacobian_matrix(dofs_per_cell,
                                                       dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);

    // allocate memory for
    // - local diffusion parameters
    std::vector<std::shared_ptr<LocalDiffusionParameters<Number>>> ldp(
      n_q_points);
    // - current solution at quadrature points
    std::vector<Vector<Number>> diffusion_values(
      n_q_points, Vector<Number>(fe_data.get_fe().components));

    // get current (locally relevant) solution
    VectorType current_relevant_solution;
    fe_data.initialize_locally_relevant_vector(current_relevant_solution);
    current_relevant_solution = current_solution;

    for (const auto &element :
         fe_data.get_dof_handler().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            element_jacobian_matrix = 0.;

            fe_values.reinit(element);
            fe_values.get_function_values(current_relevant_solution,
                                          diffusion_values);

            // get local diffusion parameters
            ldp = diffusion_parameters->get_local_diffusion_parameters(element);

            for (const auto q_point : fe_values.quadrature_point_indices())
              {
                for (const auto i : fe_values.dof_indices())
                  {
                    // get component index of i
                    const auto comp_i =
                      fe_data.get_fe().system_to_component_index(i).first;

                    for (const auto j : fe_values.dof_indices())
                      {
                        // get component index of j
                        const auto comp_j =
                          fe_data.get_fe().system_to_component_index(j).first;

                        element_jacobian_matrix(i, j) +=
                          ((ldp[q_point]->get_linear_coefficient(
                              comp_i, comp_j) + // linear terms
                            ldp[q_point]->get_quadratic_coefficient_derivative(
                              comp_i, comp_j, diffusion_values[q_point])) *
                           fe_values.shape_value(i, q_point) * // phi_i
                           fe_values.shape_value(j, q_point))  // phi_j
                          * fe_values.JxW(q_point);            // dx
                      }
                  }
              }
            // distribute local-to-global
            element->get_dof_indices(local_dof_indices);
            fe_data.get_constraints().distribute_local_to_global(
              element_jacobian_matrix, local_dof_indices, jacobian);
          }
      }

    // sum up overlapping parts
    jacobian.compress(dealii::VectorOperation::add);
  }



  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  DiffusionManager<dim, VectorType, MatrixType, Number>::
    interpolate_initial_conditions()
  {
    if (initial_condition)
      {
        dealii::VectorTools::interpolate(fe_data.get_dof_handler(),
                                         *initial_condition,
                                         solution);
        fe_data.get_constraints().distribute(solution);
      }
  }



  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  DiffusionManager<dim, VectorType, MatrixType, Number>::output_results(
    const double       time,
    const unsigned int time_step)
  {
    dealii::TimerOutput::Scope timer_section(
      computing_timer, "DiffusionManager - Output results");
    // get dof handler
    const auto &dof_handler = fe_data.get_dof_handler();

    // get distributed solution
    VectorType distributed_solution;
    fe_data.initialize_locally_relevant_vector(distributed_solution);
    distributed_solution = solution;

    // create data out object and add solution
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(distributed_solution, output_component_names);

    //// write subdomain id
    dealii::Vector<float> subdomain(
      dof_handler.get_triangulation().n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = dof_handler.get_triangulation().locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    //// write material id
    dealii::Vector<float> material_ids(
      dof_handler.get_triangulation().n_active_cells());
    unsigned int it = 0;
    for (const auto &element : dof_handler.active_cell_iterators())
      {
        material_ids(it) = element->material_id();
        ++it;
      }
    data_out.add_data_vector(material_ids, "material_id");

    dealii::DataOutBase::VtkFlags flags;
    data_out.set_flags(dealii::DataOutBase::VtkFlags(time, time_step));
    flags.write_higher_order_cells = true;
    flags.compression_level = dealii::DataOutBase::CompressionLevel::best_speed;
    flags.print_date_and_time = true;
    flags.time                = time;
    data_out.set_flags(flags);

    // get mapping degree
    data_out.build_patches(fe_data.get_mapping(), fe_data.get_fe().degree);

    data_out.write_vtu_with_pvtu_record(
      output_dir, output_filename, time_step, fe_data.get_MPI_comm(), 5);
  }


  template <int dim, typename VectorType, typename MatrixType, typename Number>
  void
  DiffusionManager<dim, VectorType, MatrixType, Number>::
    update_local_diffusion_parameters(
      const std::vector<dealii::Point<dim>>     &points,
      const Number                               time,
      const std::vector<dealii::Vector<Number>> &diffusion_values,
      const std::vector<
        std::shared_ptr<Pathways::Cells::LocalCellCollection<dim, Number>>>
        &local_cell_collection,
      std::vector<std::shared_ptr<LocalDiffusionParameters<Number>>>
        &local_diffusion_parameters)
  {
    dealii::TimerOutput::Scope timer_section(
      computing_timer, "DiffusionManager - Update diffusion parameters");

    // if local diffusion parameters are flagged as constant don't do anything
    // it suffices to check the first one
    if (local_diffusion_parameters[0]->is_constant())
      {
        pcout
          << "Local diffusion parameters marked as constant! Nothing to do..."
          << std::endl;
        return;
      }

    Assert(
      !(transformers.empty()),
      dealii::ExcMessage(
        "Cannot update diffusion parameters if transformers are empty, add transformers first!"));
    Assert(
      (local_cell_collection.size() == 1) or
        (local_cell_collection.size() == local_diffusion_parameters.size()),
      dealii::ExcMessage(
        "Cannot update diffusion parameters! Number of quadrature points do not match, the supported"
        "cases are: (local_cell_collection.size() == 1) or (local_cell_collection.size() == local_diffusion_parameters.size())"));

    // get number of local cell collections on the element
    const unsigned int n_cell_collections_per_element =
      local_cell_collection.size();

    // DEBUG
    // if (n_cell_collections_per_element == 1)
    //    pcout << "Update all elements of local_diffusion_parameters with the
    //    same values!" << std::endl;
    // else
    //    pcout << "Update elements of local_diffusion_parameters individually!"
    //    << std::endl;

    // loop over all quadrature points (i.e. local diffusion parameters)
    for (unsigned int q_point = 0; q_point < local_diffusion_parameters.size();
         ++q_point)
      {
        // get local cell collection
        const auto &loc_cell_collection = n_cell_collections_per_element == 1 ?
                                            *local_cell_collection[0] :
                                            *local_cell_collection[q_point];

        // loop over cells (cell types, not actual cells) in local cell
        // collection
        for (const auto &[cell_type, cell_vector] :
             loc_cell_collection.get_cells())
          {
            (void)cell_vector; // not needed

            // todo: is it really necessary to have a pathway to diffusion
            // transformer for every cell type? get transformer for that cell
            // type
            Assert(
              transformers.find(cell_type) != transformers.end(),
              dealii::ExcMessage(
                "Could not find a transformer in the diffusion problem for the given cell type!"
                "Make sure you add a transformer for the cell type '" +
                Pathways::Cells::CellType2string(cell_type) + "'!"));

            const auto &transformer = transformers[cell_type];

            // apply transformer
            transformer(points[q_point],
                        time,
                        diffusion_values[q_point],
                        loc_cell_collection.get_average_pathway_output(
                          cell_type),
                        loc_cell_collection.get_average_pathway_baseline_output(
                          cell_type),
                        *local_diffusion_parameters[q_point]);
          }
      }
  }

  // explicit instantiations
  template class DiffusionManager<3,
                                  dealii::TrilinosWrappers::MPI::Vector,
                                  dealii::TrilinosWrappers::SparseMatrix,
                                  double>;

} // namespace Diffusion
