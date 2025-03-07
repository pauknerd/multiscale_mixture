#pragma once

#include <deal.II/base/quadrature_point_data.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria_accessor.h>

#include <deal.II/lac/lapack_full_matrix.h>

#include <MSHCMM/common/fe_data.h>
#include <MSHCMM/mixture/growth_and_remodeling/local_mixture.h>

// NOTE: by default, all boundary_ids are set to 0 when using
// GridGenerator::cylinder_shell()! left boundary (z=0) -> boundary id 1 right
// boundary (z=length) -> boundary id 2 inner surface -> boundary id 3 outer
// surface -> boundary id 4
template <int dim>
void
colorize_cylinder_shell(dealii::Triangulation<dim> &triangulation,
                        const double                length,
                        const double                inner_radius,
                        const double                outer_radius)
{
  using namespace dealii;

  for (const auto &cell : triangulation.active_cell_iterators())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary())
        {
          const Point<dim> face_center = face->center();

          // z=0
          if (face_center[2] == 0.0)
            face->set_boundary_id(0);
          // z = L
          else if (face_center[2] == length)
            face->set_boundary_id(1);
          // inside of cylinder
          else if (std::sqrt(face_center[0] * face_center[0] +
                             face_center[1] * face_center[1]) <
                   (inner_radius + outer_radius) / 2)
            face->set_boundary_id(2);
          // outside of cylinder
          else
            face->set_boundary_id(3);
        }
}

namespace HELPERS
{
  template <int dim, typename VectorType>
  void
  initialize_locally_owned_vector(VectorType                    &vector,
                                  const dealii::DoFHandler<dim> &dof_handler_in)
  {
    // initialize based on locally owned dofs
    vector.reinit(dof_handler_in.locally_owned_dofs(),
                  dof_handler_in.get_communicator());
  }

  // overload for serial vectors
  template <int dim, typename Number = double>
  void
  initialize_locally_owned_vector(dealii::Vector<Number>        &vector,
                                  const dealii::DoFHandler<dim> &dof_handler_in)
  {
    vector.reinit(dof_handler_in.n_dofs());
  }

  /**
   * Compute the current volume of the simulation domain.
   *
   * Compute current volume of simulation domain by looping overall elements,
   * get the mixture-level deformation gradient from the local mixture
   * quadrature point data and integrating det(F_mixture) over the entire
   * domain.
   *
   * @returns current_volume
   */
  template <int dim, typename Number = double>
  Number
  compute_current_volume(const Common::FEData<dim, Number> &fe_data,
                         const dealii::CellDataStorage<
                           typename dealii::Triangulation<dim>::cell_iterator,
                           Mixture::LocalMixture<dim, Number>> &local_mixtures)
  {
    Number local_volume = 0.0;

    auto fe_values = fe_data.make_fe_values(dealii::update_JxW_values);

    for (const auto &element :
         fe_data.get_dof_handler().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            fe_values.reinit(element);

            const auto local_mixture = local_mixtures.get_data(element);

            for (const auto q_point : fe_values.quadrature_point_indices())
              {
                const auto F_mixture =
                  local_mixture[q_point]->get_mixture_deformation_gradient();
                const auto JxW = fe_values.JxW(q_point);

                local_volume += determinant(F_mixture) * JxW;
              }
          }
      }
    Assert(local_volume > 0.0, dealii::ExcInternalError());
    // return sum over all processors
    return dealii::Utilities::MPI::sum(local_volume, fe_data.get_MPI_comm());
  }


  template <int dim, typename VectorType, typename Number = double>
  Number
  get_max_nodal_displacement(const dealii::DoFHandler<dim> &dof_handler,
                             const VectorType              &total_solution)
  {
    // inspired by deal.ii step-18
    // max nodal displacement on local processor
    Number max_nodal_displacement = 0.0;
    for (const auto &element : dof_handler.active_cell_iterators())
      {
        if (element->is_locally_owned())
          for (const auto v : element->vertex_indices())
            {
              Number vertex_displacement = 0.0;
              for (unsigned int d = 0; d < dim; ++d)
                vertex_displacement +=
                  total_solution(element->vertex_dof_index(v, d)) *
                  total_solution(element->vertex_dof_index(v, d));

              vertex_displacement = std::sqrt(vertex_displacement);

              if (vertex_displacement > max_nodal_displacement)
                max_nodal_displacement = vertex_displacement;
            }
      }

    // global comparison across processors
    return dealii::Utilities::MPI::max(max_nodal_displacement,
                                       dof_handler.get_communicator());
  }

  template <int dim, typename VectorType, typename Number = double>
  Number
  get_avg_nodal_displacement(const dealii::DoFHandler<dim> &dof_handler,
                             const VectorType              &total_solution)
  {
    // from deal.ii step-18
    // problem is that some nodes are visited twice, that is why we need to keep
    // track of which vertices have been visited already
    std::vector<bool> vertex_visited(
      dof_handler.get_triangulation().n_used_vertices(), false);
    Number total_nodal_displacement = 0.0;

    for (const auto &element : dof_handler.active_cell_iterators())
      {
        if (element->is_locally_owned())
          for (const auto v : element->vertex_indices())
            if (vertex_visited[element->vertex_index(v)] == false)
              {
                vertex_visited[element->vertex_index(v)] = true;

                Number vertex_displacement = 0.0;
                for (unsigned int d = 0; d < dim; ++d)
                  vertex_displacement +=
                    total_solution(element->vertex_dof_index(v, d)) *
                    total_solution(element->vertex_dof_index(v, d));

                vertex_displacement = std::sqrt(vertex_displacement);

                // sum up individual nodal displacements
                total_nodal_displacement += vertex_displacement;
              }
      }

    // divide by number of global nodes
    // todo: check if the number of vertices is correct when using distributed
    // triangulation with MPI
    const auto avg_nodal_displacement =
      total_nodal_displacement /
      dof_handler.get_triangulation().n_used_vertices();

    // global sum across processors
    return dealii::Utilities::MPI::sum(avg_nodal_displacement,
                                       dof_handler.get_communicator());
  }

  /**
   * @brief Compute all the deformation gradients at all quadrature points at once.
   *
   * @pre Assumes F already has the same size as displacement_gradients.
   *
   * @param displacement_gradients
   * @param F
   */
  template <int dim, typename Number = double>
  void
  compute_deformation_gradients(
    const std::vector<dealii::Tensor<2, dim, Number>> &displacement_gradients,
    std::vector<dealii::Tensor<2, dim, Number>>       &F)
  {
    for (size_t i = 0; i < displacement_gradients.size(); ++i)
      F[i] =
        dealii::Physics::Elasticity::Kinematics::F(displacement_gradients[i]);
  }

  template <int dim, typename Number = double>
  std::pair<dealii::Tensor<2, dim, Number>, dealii::Tensor<2, dim, Number>>
  compute_polar_decomposition(const dealii::Tensor<2, dim, Number> &F)
  {
    // copy tensor to FullMatrix
    dealii::FullMatrix<Number> F_fm(dim);
    F_fm.copy_from(F);

    // copy tensor to LAPACKFullMatrix
    dealii::LAPACKFullMatrix<Number> F_la(dim);
    F_la.copy_from(F_fm);

    // compute SVD
    F_la.compute_svd();
    // get U, sigma, and V^T
    const auto U_svd = F_la.get_svd_u();
    // U_svd.print_formatted(std::cout);

    dealii::Vector<Number> sigma(dim);
    for (unsigned int i = 0; i < dim; i++)
      sigma(i) = F_la.singular_value(i);
    // std::cout << sigma << std::endl;

    const auto VT_svd = F_la.get_svd_vt();
    // VT_svd.print_formatted(std::cout);

    // compute R
    dealii::FullMatrix<Number> R_fm(dim);
    U_svd.mmult(R_fm, VT_svd);
    //    std::cout << "R:" << std::endl;
    //    R_fm.print_formatted(std::cout);
    // compute U
    dealii::LAPACKFullMatrix<Number> U_la(dim);
    VT_svd.Tmmult(U_la, VT_svd, sigma);
    //    std::cout << "U:" << std::endl;
    //    U_la.print_formatted(std::cout);

    // transform LAPACKFullMatrix into FullMatrix
    dealii::FullMatrix<Number> U_fm(dim);
    U_fm = U_la;

    // copy into tensors
    dealii::Tensor<2, dim> R;
    dealii::Tensor<2, dim> U;

    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        {
          R[i][j] = R_fm(i, j);
          U[i][j] = U_fm(i, j);
        }

    return std::make_pair(R, U);
  }

} // namespace HELPERS
