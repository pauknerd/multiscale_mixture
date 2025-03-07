#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/component_mask.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/vector_tools.h>

namespace Common
{
  /**
   * Different boundary types.
   */
  enum class BoundaryType
  {
    Undefined, // implicit homogenous Neumann BC
    Dirichlet,
    Neumann, // Neumann boundary condition; in case of mechanical problem in
             // reference configuration
    Pressure // only applicable to mechanical problem
  };

  /**
   * Class for managing the boundary conditions.
   */
  template <unsigned int dim, typename Number = double>
  class BoundaryDescriptor
  {
  public:
    void
    clear()
    {
      dirichlet_bc.clear();
      neumann_bc.clear();
      pressure_bc.clear();
    }

    // add dirichlet boundary, default ComponentMask applies to all components
    void
    add_dirichlet_bc(const dealii::types::boundary_id      &boundary_id,
                     std::shared_ptr<dealii::Function<dim>> function,
                     const dealii::ComponentMask           &component_mask)
    {
      dirichlet_bc.emplace(boundary_id,
                           std::make_pair(function, component_mask));
    }

    /**
     * @brief Replace a Dirichlet boundary condition on an already existing boundary.
     *
     * @details
     * Throws an error if nothing is prescribed on the given `boundary_id`,
     * i.e., if there is nothing to replace. This function only replaces, it
     * does not create.
     *
     * @param boundary_id
     * @param function
     * @param component_mask
     */
    void
    replace_dirichlet_bc(const dealii::types::boundary_id      &boundary_id,
                         std::shared_ptr<dealii::Function<dim>> function,
                         const dealii::ComponentMask           &component_mask)
    {
      dirichlet_bc[boundary_id] = std::make_pair(function, component_mask);
    }

    /**
     * @brief Add a 1st Piola-Kirchhoff (PK1) boundary condition. Note that the @param function
     * is a second-order tensor function which will be multiplied by the face
     * normal in the reference configuration.
     *
     * @param boundary_id
     * @param function
     */
    void
    add_neumann_bc(const dealii::types::boundary_id              &boundary_id,
                   std::shared_ptr<dealii::Function<dim, Number>> function)
    {
      neumann_bc.emplace(boundary_id, function);
    }

    void
    add_pressure_bc(const dealii::types::boundary_id              &boundary_id,
                    std::shared_ptr<dealii::Function<dim, Number>> function)
    {
      pressure_bc.emplace(boundary_id, function);
    }

    /**
     * @brief Replace a Neumann boundary condition on an already existing boundary.
     *
     * @details
     * Throws an error if nothing is prescribed on the given `boundary_id`,
     * i.e., if there is nothing to replace. This function only replaces, it
     * does not create.
     * @param boundary_id
     * @param function
     */
    void
    replace_neumann_bc(const dealii::types::boundary_id      &boundary_id,
                       std::shared_ptr<dealii::Function<dim>> function)
    {
      neumann_bc[boundary_id] = function;
    }

    void
    set_evaluation_time(const Number time)
    {
      // for all Dirichlet boundaries
      for (auto &i : dirichlet_bc)
        i.second.first->set_time(time);

      // for all Neumann boundaries
      for (auto &i : neumann_bc)
        i.second->set_time(time);

      // for all Pressure boundaries
      for (auto &i : pressure_bc)
        i.second->set_time(time);
    }

    [[nodiscard]] BoundaryType
    get_boundary_type(const dealii::types::boundary_id &boundary_id) const
    {
      if (this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
        return BoundaryType::Dirichlet;
      else if (this->neumann_bc.find(boundary_id) != this->neumann_bc.end())
        return BoundaryType::Neumann;
      else if (this->pressure_bc.find(boundary_id) != this->pressure_bc.end())
        return BoundaryType::Pressure;
      return BoundaryType::Undefined;
    }

    std::pair<BoundaryType, std::shared_ptr<dealii::Function<dim>>>
    get_boundary(const dealii::types::boundary_id &boundary_id) const
    {
      // get dirichlet conditions on the given boundary id
      {
        auto res = this->dirichlet_bc.find(boundary_id);
        if (res != this->dirichlet_bc.end())
          return {BoundaryType::Dirichlet, res->second.first};
      }

      // get neumann conditions on the given boundary id
      {
        auto res = this->neumann_bc.find(boundary_id);
        if (res != this->neumann_bc.end())
          return {BoundaryType::Neumann, res->second};
      }

      // get pressure conditions on the given boundary id
      {
        auto res = this->pressure_bc.find(boundary_id);
        if (res != this->pressure_bc.end())
          return {BoundaryType::Pressure, res->second};
      }

      return {BoundaryType::Undefined,
              std::shared_ptr<dealii::Function<dim>>(
                new dealii::Functions::ZeroFunction<dim>(dim))};
    }

    std::pair<BoundaryType, std::shared_ptr<dealii::Function<dim, Number>>>
    get_neumann_boundary(const dealii::types::boundary_id &boundary_id) const
    {
      // get neumann conditions on the given boundary id
      {
        auto res = this->neumann_bc.find(boundary_id);
        if (res != this->neumann_bc.end())
          return {BoundaryType::Neumann, res->second};
      }

      return {BoundaryType::Undefined,
              std::shared_ptr<dealii::Function<dim, Number>>(
                new dealii::Functions::ZeroFunction<dim, Number>(dim))};
    }

    std::pair<BoundaryType, std::shared_ptr<dealii::Function<dim, Number>>>
    get_pressure_boundary(const dealii::types::boundary_id &boundary_id) const
    {
      // get pressure boundary conditions on the given boundary id
      {
        auto res = this->pressure_bc.find(boundary_id);
        if (res != this->pressure_bc.end())
          return {BoundaryType::Pressure, res->second};
      }

      return {BoundaryType::Undefined,
              std::shared_ptr<dealii::Function<dim, Number>>(
                new dealii::Functions::ZeroFunction<dim, Number>(dim))};
    }

    // check if pressure boundary conditions are present
    [[nodiscard]] bool
    has_pressure_boundary_condition() const
    {
      return !pressure_bc.empty();
    }

    // build dirichlet conditions. Also handles the case of inhomogeneous
    // dirichlet BCs
    void
    build_dirichlet_constraints(const dealii::DoFHandler<dim>     &dof_handler,
                                dealii::AffineConstraints<Number> &constraints,
                                const bool homogeneous = false)
    {
      constraints.clear();
      // loop over all dirichlet constraints and build AffineConstraint matrix
      for (const auto &dbc : dirichlet_bc)
        {
          // build constraints but with all constrained dofs replaced by zero
          // values
          if (homogeneous)
            {
              // boundary_id
              const auto boundary_id = dbc.first;
              // function (note that it is a shared_ptr)
              const auto boundary_function =
                dealii::Functions::ZeroFunction<dim, Number>(
                  dbc.second.first->n_components);
              // component mask
              const auto component_mask = dbc.second.second;
              // apply dirichlet BC
              dealii::VectorTools::interpolate_boundary_values(
                dof_handler,
                boundary_id,
                boundary_function,
                constraints,
                component_mask);
            }
          else
            {
              // boundary_id
              const auto boundary_id = dbc.first;
              // function (note that it is a shared_ptr)
              const auto boundary_function = dbc.second.first;
              // component mask
              const auto component_mask = dbc.second.second;
              // apply dirichlet BC
              dealii::VectorTools::interpolate_boundary_values(
                dof_handler,
                boundary_id,
                *boundary_function,
                constraints,
                component_mask);
            }
        }
      constraints.close();
    }

  private:
    std::map<
      dealii::types::boundary_id,
      std::pair<std::shared_ptr<dealii::Function<dim>>, dealii::ComponentMask>>
      dirichlet_bc;
    std::map<dealii::types::boundary_id,
             std::shared_ptr<dealii::Function<dim, Number>>>
      neumann_bc;
    std::map<dealii::types::boundary_id,
             std::shared_ptr<dealii::Function<dim, Number>>>
      pressure_bc;
  };
} // namespace Common
