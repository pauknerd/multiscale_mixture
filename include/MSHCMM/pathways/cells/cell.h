#pragma once

#include "cell_state.h"
#include "cell_traits.h"

namespace Pathways::Cells
{
  /**
   * Class that represents a generic cell.
   *
   * @tparam Number
   */
  template <typename Number = double>
  class Cell
  {
  public:
    //! initialize cell state based on number of components (values will be
    //! zero)
    // todo: add random initialization?
    Cell(const CellType    &cell_type,
         const unsigned int pathway_id,
         const Number       pathway_weight,
         const unsigned int n_components_state,
         const unsigned int n_inputs,
         const unsigned int n_outputs);

    //! initialize cell state based on defined initial state
    Cell(const CellType               &cell_type,
         const unsigned int            pathway_id,
         const Number                  pathway_weight,
         const dealii::Vector<Number> &initial_state,
         const unsigned int            n_inputs,
         const unsigned int            n_outputs);

    dealii::Vector<Number> &
    get_cell_state();

    const dealii::Vector<Number> &
    get_cell_state() const;

    // todo: not sure if that is a good solution
    std::vector<std::reference_wrapper<Number>>
    get_pathway_input();

    std::vector<Number>
    get_pathway_output() const;

    void
    store_baseline_pathway_output();

    const std::vector<Number> &
    get_baseline_pathway_output() const;

    [[nodiscard]] const CellType &
    get_cell_type() const;

    [[nodiscard]] unsigned int
    get_pathway_id() const;

    [[nodiscard]] Number
    get_pathway_weight() const;

  private:
    // store general traits of the cell
    CellTraits cell_traits;
    // store the current state of the cell signaling pathway
    CellState<Number> cell_state;
    // todo: add function to compute phenotype?

    std::vector<Number> baseline_pathway_input;
    std::vector<Number> baseline_pathway_output;
  };
} // namespace Pathways::Cells
