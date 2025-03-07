#pragma once

#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <memory>
#include <utility>

namespace Common
{
  /**
   * Container class for everything related to finite elements.
   */
  template <int dim, typename Number = double>
  class FEData
  {
  public:
    /**
     * Default constructor.
     */
    FEData() = default;

    /** Initializes all relevant structures.
     *
     * @param triangulation
     * @param fe_degree
     * @param n_components
     */
    FEData(const dealii::Triangulation<dim> &triangulation,
           const unsigned int                fe_degree,
           const unsigned int                quad_degree,
           const unsigned int                n_components)
      : fe(std::make_unique<dealii::FESystem<dim>>(dealii::FE_Q<dim>(fe_degree),
                                                   n_components))
      , quadrature(std::make_unique<dealii::QGauss<dim>>(quad_degree))
      , face_quadrature(std::make_unique<dealii::QGauss<dim - 1>>(quad_degree))
      , dof_handler(std::make_unique<dealii::DoFHandler<dim>>(triangulation))
      , constraints(std::make_unique<dealii::AffineConstraints<Number>>())
      , zero_constraints(std::make_unique<dealii::AffineConstraints<Number>>())
      , mapping(std::make_unique<dealii::MappingQ<dim>>(fe_degree))
    {
      internal_setup();
    }

    FEData(FEData<dim, Number> &&src) noexcept
    {
      // move unique pointers and IndexSets
      move_from(src);
      // sparsity patterns don't support move semantics, so they need to be
      // setup again
      initialize_sparsity_patterns();
    }

    FEData<dim, Number> &
    operator=(FEData<dim, Number> &&rhs) noexcept
    {
      // check for self-assignment
      if (this == &rhs)
        return *this;
      // move unique pointers and IndexSets
      move_from(rhs);
      // sparsity patterns don't support move semantics, so they need to be
      // setup again
      initialize_sparsity_patterns();

      return *this;
    }

    void
    reinit(const dealii::Triangulation<dim> &triangulation,
           const unsigned int                fe_degree,
           const unsigned int                quad_degree,
           const unsigned int                n_components)
    {
      fe = std::make_unique<dealii::FESystem<dim>>(dealii::FE_Q<dim>(fe_degree),
                                                   n_components);
      quadrature      = std::make_unique<dealii::QGauss<dim>>(quad_degree);
      face_quadrature = std::make_unique<dealii::QGauss<dim - 1>>(quad_degree);
      dof_handler = std::make_unique<dealii::DoFHandler<dim>>(triangulation);
      constraints = std::make_unique<dealii::AffineConstraints<Number>>();
      zero_constraints = std::make_unique<dealii::AffineConstraints<Number>>();
      mapping          = std::make_unique<dealii::MappingQ<dim>>(fe_degree);

      internal_setup();
    }

    const dealii::FESystem<dim> &
    get_fe() const
    {
      return *fe;
    }

    const dealii::Triangulation<dim> &
    get_triangulation() const
    {
      return dof_handler->get_triangulation();
    }

    // swap FESystem with an arbitrary new FESystem and redistribute dofs
    void
    swap_fe(std::unique_ptr<dealii::FESystem<dim>> new_fe_system)
    {
      fe.swap(new_fe_system);
      // since FESystem has changed the internal objects have to be
      // reinitialized to match the new number of DoFs.
      internal_setup();
    }

    const dealii::Quadrature<dim> &
    get_quadrature() const
    {
      return *quadrature;
    }

    const dealii::Quadrature<dim - 1> &
    get_face_quadrature() const
    {
      return *face_quadrature;
    }

    const dealii::DoFHandler<dim> &
    get_dof_handler() const
    {
      return *dof_handler;
    }

    const dealii::AffineConstraints<Number> &
    get_constraints() const
    {
      return *constraints;
    }

    dealii::AffineConstraints<Number> &
    get_constraints()
    {
      return *constraints;
    }

    const dealii::AffineConstraints<Number> &
    get_zero_constraints() const
    {
      return *zero_constraints;
    }

    dealii::AffineConstraints<Number> &
    get_zero_constraints()
    {
      return *zero_constraints;
    }

    const dealii::Mapping<dim> &
    get_mapping() const
    {
      return *mapping;
    }

    // create FEValues with class quadrature rule
    dealii::FEValues<dim>
    make_fe_values(const dealii::UpdateFlags update_flags) const
    {
      return dealii::FEValues<dim>(*mapping, *fe, *quadrature, update_flags);
    }

    // create FEValues with custom quadrature rule
    dealii::FEValues<dim>
    make_fe_values(const dealii::Quadrature<dim> &custom_quadrature,
                   const dealii::UpdateFlags      update_flags) const
    {
      return dealii::FEValues<dim>(*mapping,
                                   *fe,
                                   custom_quadrature,
                                   update_flags);
    }

    dealii::FEFaceValues<dim>
    make_fe_face_values(const dealii::UpdateFlags update_flags) const
    {
      return dealii::FEFaceValues<dim>(*mapping,
                                       *fe,
                                       *face_quadrature,
                                       update_flags);
    }

    // create FEFaceValues with custom quadrature rule
    dealii::FEFaceValues<dim>
    make_fe_face_values(
      const dealii::Quadrature<dim - 1> &custom_face_quadrature,
      const dealii::UpdateFlags          update_flags) const
    {
      return dealii::FEFaceValues<dim>(*mapping,
                                       *fe,
                                       custom_face_quadrature,
                                       update_flags);
    }

    MPI_Comm
    get_MPI_comm() const
    {
      return dof_handler->get_communicator();
    }

    template <typename VectorType>
    void
    initialize_locally_owned_vector(VectorType &vector) const
    {
      vector.reinit(locally_owned_dofs, get_MPI_comm());
    }

    // overload for serial vectors
    void
    initialize_locally_owned_vector(dealii::Vector<Number> &vector) const
    {
      vector.reinit(dof_handler->n_dofs());
    }

    template <typename VectorType>
    void
    initialize_locally_relevant_vector(VectorType &vector) const
    {
      vector.reinit(locally_owned_dofs, locally_relevant_dofs, get_MPI_comm());
    }

    // overload for serial vectors
    void
    initialize_locally_relevant_vector(dealii::Vector<Number> &vector) const
    {
      vector.reinit(dof_handler->n_dofs());
    }

    // initialize a vector with a different DoFHandler
    template <typename VectorType>
    void
    initialize_locally_owned_vector(
      VectorType                    &vector,
      const dealii::DoFHandler<dim> &dof_handler_in) const
    {
      // initialize based on locally owned dofs
      vector.reinit(dof_handler_in.locally_owned_dofs(), get_MPI_comm());
    }

    // overload for serial vectors
    void
    initialize_locally_owned_vector(
      dealii::Vector<Number>        &vector,
      const dealii::DoFHandler<dim> &dof_handler_in) const
    {
      vector.reinit(dof_handler_in.n_dofs());
    }

    // initialize a vector with a different DoFHandler
    template <typename VectorType>
    void
    initialize_locally_relevant_vector(
      VectorType                    &vector,
      const dealii::DoFHandler<dim> &dof_handler_in) const
    {
      dealii::IndexSet locally_relevant_dofs_temp;
      dealii::DoFTools::extract_locally_relevant_dofs(
        dof_handler_in, locally_relevant_dofs_temp);
      // initialize based on locally owned dofs
      vector.reinit(dof_handler_in.locally_owned_dofs(),
                    locally_relevant_dofs_temp,
                    get_MPI_comm());
    }

    // overload for serial vectors
    void
    initialize_locally_relevant_vector(
      dealii::Vector<Number>        &vector,
      const dealii::DoFHandler<dim> &dof_handler_in) const
    {
      vector.reinit(dof_handler_in.n_dofs());
    }

    template <typename MatrixType>
    void
    initialize_matrix(MatrixType &matrix) const
    {
      matrix.reinit(locally_owned_dofs,
                    locally_owned_dofs,
                    dsp,
                    get_MPI_comm());
    }

    // overload for serial matrix
    void
    initialize_matrix(dealii::SparseMatrix<Number> &matrix) const
    {
      sparsity_pattern.copy_from(dsp);

      matrix.reinit(sparsity_pattern);
    }

    const dealii::IndexSet &
    get_locally_relevant_dofs() const
    {
      return locally_relevant_dofs;
    }

    const dealii::IndexSet &
    get_locally_owned_dofs() const
    {
      return locally_owned_dofs;
    }

    enum
    {
      u_dof = 0
    };

  private:
    void
    move_from(FEData<dim, Number> &src) noexcept
    {
      // move unique pointers
      fe               = std::move(src.fe);
      quadrature       = std::move(src.quadrature);
      face_quadrature  = std::move(src.face_quadrature);
      dof_handler      = std::move(src.dof_handler);
      constraints      = std::move(src.constraints);
      zero_constraints = std::move(src.zero_constraints);
      mapping          = std::move(src.mapping);
      // move IndexSets
      locally_owned_dofs    = std::move(src.locally_owned_dofs);
      locally_relevant_dofs = std::move(src.locally_relevant_dofs);
    }

    void
    internal_setup()
    {
      // distribute DoFs
      dof_handler->distribute_dofs(*fe);

      // renumber dofs
      dealii::DoFRenumbering::Cuthill_McKee(*dof_handler);

      // extract DoF information
      locally_owned_dofs = dof_handler->locally_owned_dofs();
      dealii::DoFTools::extract_locally_relevant_dofs(*dof_handler,
                                                      locally_relevant_dofs);

      // initialize constraints
      constraints->clear();
      constraints->reinit(locally_relevant_dofs);
      zero_constraints->reinit(locally_relevant_dofs);

      // NOTE: not sure if this is needed
      dealii::DoFTools::make_hanging_node_constraints(*dof_handler,
                                                      *constraints);

      initialize_sparsity_patterns();
    }

    void
    initialize_sparsity_patterns() noexcept
    {
      // setup sparsity pattern
      dsp.reinit(locally_relevant_dofs.size(),
                 locally_relevant_dofs.size(),
                 locally_relevant_dofs);

      dealii::DoFTools::make_sparsity_pattern(*dof_handler,
                                              dsp,
                                              *zero_constraints,
                                              false);
      dealii::SparsityTools::distribute_sparsity_pattern(
        dsp,
        dof_handler->locally_owned_dofs(),
        dof_handler->get_communicator(),
        locally_relevant_dofs);
    }

    std::unique_ptr<dealii::FESystem<dim>>             fe;
    std::unique_ptr<dealii::Quadrature<dim>>           quadrature;
    std::unique_ptr<dealii::Quadrature<dim - 1>>       face_quadrature;
    std::unique_ptr<dealii::DoFHandler<dim>>           dof_handler;
    std::unique_ptr<dealii::AffineConstraints<Number>> constraints;
    std::unique_ptr<dealii::AffineConstraints<Number>> zero_constraints;
    std::unique_ptr<dealii::Mapping<dim>>              mapping;

    dealii::IndexSet locally_owned_dofs;
    dealii::IndexSet locally_relevant_dofs;

    dealii::DynamicSparsityPattern dsp;
    // needed for serial matrix, needs to be mutable so that it can be modified
    // in const function initialize_matrix(SparseMatrix<Number>)
    mutable dealii::SparsityPattern sparsity_pattern;
  };
} // namespace Common
