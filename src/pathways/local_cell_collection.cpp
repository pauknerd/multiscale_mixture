#include <MSHCMM/pathways/cells/local_cell_collection.h>

namespace Pathways::Cells
{
  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    void
    LocalCellCollection<dim, Number>::add_cell(const Cell<Number> &cell)
  {
    // cells.emplace_back(cell);
    cells[cell.get_cell_type()].emplace_back(cell);
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    void
    LocalCellCollection<dim, Number>::add_cells(
      const CellType            &cell_type,
      const std::vector<Number> &pathway_weights,
      const unsigned int         n_components_state,
      const unsigned int         n_inputs,
      const unsigned int         n_outputs)
  {
    // create vector of pathway ids. Ids are distributed based on the order of
    // addition to the pathway_storage
    std::vector<unsigned int> pathway_ids(pathway_weights.size());
    std::iota(pathway_ids.begin(), pathway_ids.end(), 0);

    for (auto &pathway_id : pathway_ids)
      cells[cell_type].emplace_back(cell_type,
                                    pathway_id,
                                    pathway_weights[pathway_id],
                                    n_components_state,
                                    n_inputs,
                                    n_outputs);
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    std::unordered_map<CellType, std::vector<Cell<Number>>> &
    LocalCellCollection<dim, Number>::get_cells()
  {
    return cells;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    const std::unordered_map<CellType, std::vector<Cell<Number>>>       &
    LocalCellCollection<dim, Number>::get_cells() const
  {
    return cells;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    const std::vector<Cell<Number>> &
    LocalCellCollection<dim, Number>::get_cell(const CellType &cell_type) const
  {
    return cells.at(cell_type);
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    std::vector<Cell<Number>> &
    LocalCellCollection<dim, Number>::get_cell(const CellType &cell_type)
  {
    return cells.at(cell_type);
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    unsigned int
    LocalCellCollection<dim, Number>::n_cells() const
  {
    unsigned int n_cells = 0;
    for (const auto &[cell_type, cell_vector] : cells)
      n_cells += cell_vector.size();

    return n_cells;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    unsigned int
    LocalCellCollection<dim, Number>::max_pathway_components(
      const CellType &cell_type) const
  {
    const auto &cell_vector = cells.at(cell_type);
    // find cell with max number of pathway components for given cell type
    const auto it = std::max_element(std::cbegin(cell_vector),
                                     std::cend(cell_vector),
                                     [](const auto &a, const auto &b) {
                                       return a.get_cell_state().size() <
                                              b.get_cell_state().size();
                                     });

    // return the cell state size of that cell
    return it->get_cell_state().size();
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    unsigned int
    LocalCellCollection<dim, Number>::max_pathway_components() const
  {
    std::vector<unsigned int> max_pw_components_per_cell_type(cells.size());
    // find max number of pathway components for each cell type
    for (const auto &[cell_type, cell_vector] : cells)
      max_pw_components_per_cell_type.push_back(
        max_pathway_components(cell_type));
    // find max element in that vector
    const auto it =
      std::max_element(std::cbegin(max_pw_components_per_cell_type),
                       std::cend(max_pw_components_per_cell_type),
                       [](const auto &a, const auto &b) { return a < b; });

    // return the cell state size of that cell
    return *it;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    std::vector<Number>
    LocalCellCollection<dim, Number>::get_average_pathway_output(
      const CellType &cell_type) const
  {
    // initialize vector based on first cell in vector of cells. That assumes
    // that all the pathways of the same cell type have the same number of
    // output components.
    const auto n_output_components =
      cells.at(cell_type)[0].get_pathway_output().size();
    std::vector<Number> average_pathway_output(n_output_components);
    // create temporary values
    Number              weight = 0.0;
    std::vector<Number> pathway_output(n_output_components);

    for (const auto &cell : cells.at(cell_type))
      {
        weight         = cell.get_pathway_weight();
        pathway_output = cell.get_pathway_output();
        for (unsigned int i = 0; i < n_output_components; ++i)
          average_pathway_output[i] += weight * pathway_output[i];
      }

    return average_pathway_output;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    std::vector<Number>
    LocalCellCollection<dim, Number>::get_average_pathway_baseline_output(
      const CellType &cell_type) const
  {
    // initialize vector based on first cell in vector of cells. That assumes
    // that all the pathways of the same cell type have the same number of
    // output components.
    const auto n_output_components =
      cells.at(cell_type)[0].get_pathway_output().size();
    std::vector<Number> average_pathway_baseline_output(n_output_components);
    // create temporary values
    Number              weight = 0.0;
    std::vector<Number> baseline_pathway_output(n_output_components);

    for (const auto &cell : cells.at(cell_type))
      {
        weight                  = cell.get_pathway_weight();
        baseline_pathway_output = cell.get_baseline_pathway_output();

        for (unsigned int i = 0; i < n_output_components; ++i)
          average_pathway_baseline_output[i] +=
            weight * baseline_pathway_output[i];
      }

    return average_pathway_baseline_output;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    void
    LocalCellCollection<dim, Number>::set_location(
      const dealii::Point<dim> &location)
  {
    location_ = location;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    const dealii::Point<dim>       &
    LocalCellCollection<dim, Number>::get_location() const
  {
    return location_;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    bool
    LocalCellCollection<dim, Number>::cell_type_exists(
      const Pathways::Cells::CellType &cell_type) const
  {
    return cells.find(cell_type) != cells.end();
  }

  // instantiations
  template class LocalCellCollection<2, double>;
  template class LocalCellCollection<3, double>;
} // namespace Pathways::Cells