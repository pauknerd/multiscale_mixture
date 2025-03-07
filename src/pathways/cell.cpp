#include <MSHCMM/pathways/cells/cell.h>

namespace Pathways::Cells
{

  template <typename Number>
  Cell<Number>::Cell(const CellType    &cell_type,
                     const unsigned int pathway_id,
                     const Number       pathway_weight,
                     const unsigned int n_components_state,
                     const unsigned int n_inputs,
                     const unsigned int n_outputs)
    : cell_traits(cell_type, pathway_id, pathway_weight)
    , cell_state(n_components_state, n_inputs, n_outputs)
  {}

  template <typename Number>
  Cell<Number>::Cell(const CellType               &cell_type,
                     const unsigned int            pathway_id,
                     const Number                  pathway_weight,
                     const dealii::Vector<Number> &initial_state,
                     const unsigned int            n_inputs,
                     const unsigned int            n_outputs)
    : cell_traits(cell_type, pathway_id, pathway_weight)
    , cell_state(initial_state, n_inputs, n_outputs)
  {}

  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Vector<Number> &
    Cell<Number>::get_cell_state()
  {
    return cell_state.cell_state;
  }

  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    const dealii::Vector<Number>       &
    Cell<Number>::get_cell_state() const
  {
    return cell_state.cell_state;
  }

  // todo: not sure if that is a good solution
  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    std::vector<std::reference_wrapper<Number>>
    Cell<Number>::get_pathway_input()
  {
    return std::vector<std::reference_wrapper<Number>>(
      get_cell_state().begin(),
      get_cell_state().begin() + (cell_state.n_inputs));
  }

  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    std::vector<Number>
    Cell<Number>::get_pathway_output() const
  {
    return std::vector<Number>(get_cell_state().end() - (cell_state.n_outputs),
                               get_cell_state().end());
  }

  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    void
    Cell<Number>::store_baseline_pathway_output()
  {
    baseline_pathway_output = get_pathway_output();
  }

  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    const std::vector<Number>       &
    Cell<Number>::get_baseline_pathway_output() const
  {
    return baseline_pathway_output;
  }

  template <typename Number>
  [[nodiscard]] inline DEAL_II_ALWAYS_INLINE //
    const CellType &
    Cell<Number>::get_cell_type() const
  {
    return cell_traits.cell_type;
  }

  template <typename Number>
  [[nodiscard]] inline DEAL_II_ALWAYS_INLINE //
    unsigned int
    Cell<Number>::get_pathway_id() const
  {
    return cell_traits.pathway_id;
  }

  template <typename Number>
  [[nodiscard]] inline DEAL_II_ALWAYS_INLINE //
    Number
    Cell<Number>::get_pathway_weight() const
  {
    return cell_traits.pathway_weight;
  }

  // instantiations
  template class Cell<double>;
} // namespace Pathways::Cells
