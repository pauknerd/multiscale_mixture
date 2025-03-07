#pragma once

#include <deal.II/base/point.h>

#include <unordered_map>
#include <vector>

#include "cell.h"

namespace Pathways::Cells
{
  /**
   * This class handles the collection of cells on a single quadrature point.
   * Inputs of different kind are passed to this class and then distributed to
   * the pathways that use them. The distribution is done by the Transformer
   * class.
   */
  template <int dim, typename Number = double>
  class LocalCellCollection
  {
  public:
    void
    add_cell(const Cell<Number> &cell);

    void
    add_cells(const CellType            &cell_type,
              const std::vector<Number> &pathway_weights,
              const unsigned int         n_components_state,
              const unsigned int         n_inputs,
              const unsigned int         n_outputs);

    std::unordered_map<CellType, std::vector<Cell<Number>>> &
    get_cells();

    const std::unordered_map<CellType, std::vector<Cell<Number>>> &
    get_cells() const;

    const std::vector<Cell<Number>> &
    get_cell(const CellType &cell_type) const;

    std::vector<Cell<Number>> &
    get_cell(const CellType &cell_type);

    [[nodiscard]] unsigned int
    n_cells() const;

    // todo: Is that really necessary? Shouldn't the same cell type always have
    // the same size?
    [[nodiscard]] unsigned int
    max_pathway_components(const CellType &cell_type) const;

    [[nodiscard]] unsigned int
    max_pathway_components() const;

    std::vector<Number>
    get_average_pathway_output(const CellType &cell_type) const;

    std::vector<Number>
    get_average_pathway_baseline_output(const CellType &cell_type) const;

    // todo: remove location since the coordinates are passed to the
    // transformers?
    void
    set_location(const dealii::Point<dim> &location);

    const dealii::Point<dim> &
    get_location() const;

    [[nodiscard]] bool
    cell_type_exists(const CellType &cell_type) const;

  private:
    // todo: maybe use (unordered) map instead? because of accessing in
    // diffusion_update_values() std::vector<Cell<Number>> cells;
    std::unordered_map<CellType, std::vector<Cell<Number>>> cells;

    dealii::Point<dim> location_;
  };
} // namespace Pathways::Cells
