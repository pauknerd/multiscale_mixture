#pragma once

#include <deal.II/base/point.h>

#include <deal.II/grid/tria.h>

#include <MSHCMM/pathways/cells/cell_types.h>
#include <MSHCMM/pathways/equations/pathway_equation_base.h>

#include <iostream>
#include <random>
#include <vector>

/**
 * Endothelial Cell Layer.
 * Restrictions:
 * - Only supports one boundary id
 * - only one face of a single finite element can belong to that endothelial
 * cell layer!
 */


namespace Pathways
{
  //! Define the transformer for the endothelial pathway inputs.
  template <int dim, typename Number = double>
  using InputTransformerEndothelialPathway =
    std::function<void(const dealii::Point<dim>     &p,
                       const Number                  time,
                       const dealii::Vector<Number> &displacements,
                       const dealii::Vector<Number> &diffusion_values,
                       dealii::Vector<Number>       &cell_state)>;

  //! Define the transformer for the endothelial pathway output.
  template <int dim, typename Number = double>
  using OutputTransformerEndothelialPathway = std::function<
    void(const std::vector<Number> &average_pathway_output,
         const std::vector<Number> &average_baseline_pathway_output,
         dealii::Vector<Number>    &diffusion_values)>;

  //! Typedef of cell iterator.
  template <int dim>
  using CellIteratorType = typename dealii::Triangulation<dim>::cell_iterator;

  // Create FilterIterator to get the elements at the given boundary id
  class BoundaryIdEqualTo
  {
  public:
    explicit BoundaryIdEqualTo(dealii::types::boundary_id boundary_id)
      : boundary_id(boundary_id)
    {}

    template <class Iterator>
    bool
    operator()(const Iterator &i) const
    {
      if (i->is_locally_owned() && i->at_boundary())
        {
          for (const auto &face : i->face_iterators())
            {
              if (!face->at_boundary())
                continue;
              if ((face->boundary_id() == boundary_id))
                return true;
            }
        }
      return false;
    }

  protected:
    const dealii::types::boundary_id boundary_id;
  };

  template <int dim, typename Number = double>
  class EndothelialCellLayer
  {
  public:
    EndothelialCellLayer(
      const Cells::CellType &cell_type,
      std::vector<std::shared_ptr<Equations::PathwayEquationBase<Number>>>
        &&pathway_equations,
      InputTransformerEndothelialPathway<dim, Number>
        &&endothelial_input_transformer,
      OutputTransformerEndothelialPathway<dim, Number>
                       &&endothelial_output_transformer,
      const unsigned int boundary_id);

    [[nodiscard]] unsigned int
    get_endothelium_boundary_id() const;

    const Equations::ODE<Number> &
    get_pathway_equation(const unsigned int pathway_id) const;

    const InputTransformerEndothelialPathway<dim, Number> &
    get_input_transformer() const;

    const OutputTransformerEndothelialPathway<dim, Number> &
    get_output_transformer() const;

    const Cells::CellType &
    get_endothelial_cell_type() const;

    const std::vector<std::shared_ptr<Equations::PathwayEquationBase<Number>>>
    get_endothelial_pathways() const;

  private:
    // cell type, should be endothelial cell (EC)
    Cells::CellType cell_type;
    // Pathway equations
    std::vector<std::shared_ptr<Equations::PathwayEquationBase<Number>>>
      endothelial_pathways;
    // transformer
    InputTransformerEndothelialPathway<dim, Number> input_transformer;
    // transformer
    OutputTransformerEndothelialPathway<dim, Number> output_transformer;
    // boundary id of the endothelium
    unsigned int boundary_id;
  };

} // namespace Pathways
