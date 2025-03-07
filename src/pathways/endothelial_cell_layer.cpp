#include <MSHCMM/pathways/endothelial_cell_layer.h>

namespace Pathways
{

  template <int dim, typename Number>
  EndothelialCellLayer<dim, Number>::EndothelialCellLayer(
    const Cells::CellType &cell_type,
    std::vector<std::shared_ptr<Equations::PathwayEquationBase<Number>>>
      &&pathway_equations,
    InputTransformerEndothelialPathway<dim, Number>
      &&endothelial_input_transformer,
    OutputTransformerEndothelialPathway<dim, Number>
                     &&endothelial_output_transformer,
    const unsigned int boundary_id)
    : cell_type(cell_type)
    , endothelial_pathways(std::move(pathway_equations))
    , input_transformer(std::move(endothelial_input_transformer))
    , output_transformer(std::move(endothelial_output_transformer))
    , boundary_id(boundary_id){};

  template <int dim, typename Number>
  [[nodiscard]] unsigned int
  EndothelialCellLayer<dim, Number>::get_endothelium_boundary_id() const
  {
    return boundary_id;
  }

  template <int dim, typename Number>
  const Equations::ODE<Number> &
  EndothelialCellLayer<dim, Number>::get_pathway_equation(
    const unsigned int pathway_id) const
  {
    return endothelial_pathways[pathway_id]->get_ODE();
  }

  template <int dim, typename Number>
  const InputTransformerEndothelialPathway<dim, Number> &
  EndothelialCellLayer<dim, Number>::get_input_transformer() const
  {
    return input_transformer;
  }

  template <int dim, typename Number>
  const OutputTransformerEndothelialPathway<dim, Number> &
  EndothelialCellLayer<dim, Number>::get_output_transformer() const
  {
    return output_transformer;
  }

  template <int dim, typename Number>
  const Cells::CellType &
  EndothelialCellLayer<dim, Number>::get_endothelial_cell_type() const
  {
    return cell_type;
  }

  template <int dim, typename Number>
  const std::vector<std::shared_ptr<Equations::PathwayEquationBase<Number>>>
  EndothelialCellLayer<dim, Number>::get_endothelial_pathways() const
  {
    return endothelial_pathways;
  }

  // instantiations
  template class EndothelialCellLayer<2, double>;
  template class EndothelialCellLayer<3, double>;

} // namespace Pathways