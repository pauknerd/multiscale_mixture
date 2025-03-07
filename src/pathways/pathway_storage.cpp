#include <MSHCMM/pathways/pathway_storage.h>

namespace Pathways
{

  //! check if storage is empty.
  template <int dim, typename Number>
  [[nodiscard]] bool
  PathwayStorage<dim, Number>::is_empty() const
  {
    return pathway_storage.empty();
  }

  template <int dim, typename Number>
  void
  PathwayStorage<dim, Number>::add_pathways_and_transformers(
    const Cells::CellType &cell_type,
    std::vector<std::shared_ptr<Equations::PathwayEquationBase<Number>>>
                                         &&pathway_equations,
    const std::vector<Number>             &pathway_weights_vector,
    InputTransformerPathway<dim, Number> &&transformer,
    const unsigned int                     constituent_id,
    const std::vector<unsigned int>       &material_ids,
    CellLocalizer<dim>                   &&cell_localizer)
  {
    // size of pathway_equations must be equal to size of pathway_weights
    Assert(pathway_equations.size() == pathway_weights_vector.size(),
           dealii::ExcMessage(
             "Number of weights must match the number of pathways!"));

    // sum of weights must equal 1
    Assert(std::accumulate(pathway_weights_vector.cbegin(),
                           pathway_weights_vector.cend(),
                           0.0) == 1.0,
           dealii::ExcMessage(
             "Pathway weights must sum up to 1! The sum is " +
             std::to_string(std::accumulate(pathway_weights_vector.cbegin(),
                                            pathway_weights_vector.cend(),
                                            0.0)) +
             " for cell type " + Cells::CellType2string(cell_type) + "."));

    // insert pathways
    pathway_storage.emplace(cell_type, std::move(pathway_equations));

    // insert transformers
    transformers.emplace(
      cell_type, std::move(std::make_pair(transformer, constituent_id)));

    // insert pathway weights
    pathway_weights.emplace(cell_type, pathway_weights_vector);

    // insert localizer_material_id
    localizers_material_id.emplace(cell_type, material_ids);

    // insert cell_localizer
    localizers.emplace(cell_type, std::move(cell_localizer));
  }

  //! Get entire pathway_storage.
  template <int dim, typename Number>
  [[nodiscard]] const std::unordered_map<
    Cells::CellType,
    std::vector<std::shared_ptr<Equations::PathwayEquationBase<Number>>>> &
  PathwayStorage<dim, Number>::get_pathway_storage() const
  {
    return pathway_storage;
  }

  //! return the PathwayEquation associated with a given @p cell_type and @p pathway_id.
  template <int dim, typename Number>
  inline const Equations::ODE<Number> &
  PathwayStorage<dim, Number>::get_pathway_equation(
    const Cells::CellType &cell_type,
    const unsigned int     pathway_id) const
  {
    return (pathway_storage.at(cell_type))[pathway_id]->get_ODE();
  }

  //! return the PathwayEquation associated with a given @p cell_type and @p pathway_id.
  template <int dim, typename Number>
  inline const std::vector<Number> &
  PathwayStorage<dim, Number>::get_pathway_weights(
    const Cells::CellType &cell_type) const
  {
    return pathway_weights.at(cell_type);
  }

  /**
   * Return a tuple of vectors of transformers. First element are the input
   * transformers from diffusion values to pathway, second element are the
   * transformers from HCMM to the pathway.
   */
  template <int dim, typename Number>
  [[nodiscard]] inline const TransformerPair<dim, Number> &
  PathwayStorage<dim, Number>::get_transformer(
    const Cells::CellType &cell_type) const
  {
    return transformers.at(cell_type);
  }

  //! Return the cell_localizer for the given cell_type
  template <int dim, typename Number>
  [[nodiscard]] const CellLocalizer<dim> &
  PathwayStorage<dim, Number>::get_cell_localizer(
    const Cells::CellType &cell_type) const
  {
    return localizers.at(cell_type);
  }

  //! Return the material_id where the given cell_type is located
  template <int dim, typename Number>
  [[nodiscard]] const std::vector<unsigned int> &
  PathwayStorage<dim, Number>::get_cell_type_material_id(
    const Cells::CellType &cell_type) const
  {
    return localizers_material_id.at(cell_type);
  }

  template <int dim, typename Number>
  void
  PathwayStorage<dim, Number>::print_insertion_status(/*It it,*/ bool success)
  {
    std::cout << "Insertion into PathwayStorage "
              << /*it->first <<*/ (success ? "succeeded\n" : "failed\n");
  }

  template <int dim, typename Number>
  void
  PathwayStorage<dim, Number>::add_endothelial_cell_layer(
    const Cells::CellType &cell_type,
    std::vector<std::shared_ptr<Equations::PathwayEquationBase<Number>>>
                             &&endothelial_pathway_equations,
    const std::vector<Number> &endothelial_pathway_weights,
    InputTransformerEndothelialPathway<dim, Number>
      &&endothelial_input_transformer,
    OutputTransformerEndothelialPathway<dim, Number>
                       &&endothelial_output_transformer,
    const unsigned int   boundary_id,
    CellLocalizer<dim> &&cell_localizer)
  {
    // size of pathway_equations must be equal to size of pathway_weights
    Assert(endothelial_pathway_equations.size() ==
             endothelial_pathway_weights.size(),
           dealii::ExcMessage(
             "Number of weights must match the number of pathways!"));

    // sum of weights must equal 1
    Assert(std::accumulate(endothelial_pathway_weights.cbegin(),
                           endothelial_pathway_weights.cend(),
                           0.0) == 1.0,
           dealii::ExcMessage(
             "Pathway weights must sum up to 1! The sum is " +
             std::to_string(
               std::accumulate(endothelial_pathway_weights.cbegin(),
                               endothelial_pathway_weights.cend(),
                               0.0)) +
             " for cell type " + Cells::CellType2string(cell_type) + "."));

    // create endothelial layer
    endothelial_cell_layer =
      std::make_shared<EndothelialCellLayer<dim, Number>>(
        cell_type,
        std::move(endothelial_pathway_equations),
        std::move(endothelial_input_transformer),
        std::move(endothelial_output_transformer),
        boundary_id);

    // insert number of cells
    pathway_weights.emplace(cell_type, endothelial_pathway_weights);

    // insert cell_localizer
    localizers.emplace(cell_type, std::move(cell_localizer));
  }

  template <int dim, typename Number>
  inline const std::shared_ptr<EndothelialCellLayer<dim, Number>>
  PathwayStorage<dim, Number>::get_endothelial_cell_layer() const
  {
    return endothelial_cell_layer;
  }

  template <int dim, typename Number>
  std::shared_ptr<EndothelialCellLayer<dim, Number>>
  PathwayStorage<dim, Number>::get_endothelial_cell_layer()
  {
    return endothelial_cell_layer;
  }

  // instantiations
  template class PathwayStorage<2, double>;
  template class PathwayStorage<3, double>;
} // namespace Pathways
