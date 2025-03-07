#pragma once

#include <deal.II/base/point.h>

#include <MSHCMM/mixture/constituents/transferable_parameters.h>
#include <MSHCMM/pathways/cells/cell_types.h>
#include <MSHCMM/pathways/endothelial_cell_layer.h>
#include <MSHCMM/pathways/equations/pathway_equation_base.h>
#include <MSHCMM/pathways/id_distributions.h>

#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <unordered_map>
#include <utility>

namespace Pathways
{
  //! Define the transformer for the pathway inputs.
  // todo: check if argument pathway_id is needed at all
  //  -> needed if pathways of same cell type might have different scalers,
  //  e.g., different input sensitivities
  /**
   * @brief This typedef defines the interface of a transformer for pathway inputs of a specific cell type.
   *
   * @details
   * Note that you have full control over the cell state! Make sure you modify
   * the right entries to set the correct input values. Setting the wrong values
   * can be difficult to debug. However, this full control can also be used to
   * simulate the addition of a drug at a certain time that, for example, blocks
   * a specific node in the pathway equation. Note that the parameters `p` and
   * `time` can be used to mock pathway inputs and to do something as described
   * above. The `pathway_id` argument can be used to distinguish between
   * pathways of the same cell type. This can for example be used to set
   * different scalers for the different pathway equations to change sensitivity
   * to the inputs.
   */
  template <int dim, typename Number = double>
  using InputTransformerPathway = std::function<
    void(const dealii::Point<dim>     &p,
         const Number                  time,
         const dealii::Vector<Number> &diffusion_values,
         const Mixture::Constituents::TransferableParameters<Number>
                                                       &transferable_parameters,
         const unsigned int                             pathway_id,
         const dealii::Tensor<2, dim, Number>          &deformation_gradient,
         const dealii::SymmetricTensor<2, dim, Number> &PK2_stress,
         dealii::Vector<Number>                        &cell_state)>;

  /**
   * @brief Typdef for a pair of an InputTransformerPathway and a constituent id in case the pathway has a stress input from a constituent.
   */
  template <int dim, typename Number = double>
  using TransformerPair =
    std::pair<InputTransformerPathway<dim, Number>, unsigned int>;

  /**
   * @brief Function used to specify the location of cells.
   *
   * @details
   * Note that this could be problematic in case single cell collection is used.
   *
   * @param[in] q_point Quadrature point coordinates
   *
   * @returns bool True if cell should be placed here, false otherwise.
   */
  template <int dim>
  using CellLocalizer = std::function<bool(const dealii::Point<dim> &q_point)>;

  /**
   * This class takes care of storing the PathwayEquations used in the
   * simulation and the associated transformers. For each CellType, different
   * PathwayEquations can be stored, however, they can only differ in their
   * parameter values, not in their structure, i.e. the number of nodes. Note
   * that it assumed that all pathways of the same type use the same
   * transformers. Additionally, it assumed that all PathwayEquations have some
   * diffusion input whereas the stress input is optional.
   */
  template <int dim, typename Number = double>
  class PathwayStorage
  {
  public:
    //! check if storage is empty.
    [[nodiscard]] bool
    is_empty() const;

    /**
     * @brief Add `pathways` and associated `transformer` for the respective `cell_type`.
     *
     *  Add a vector of `pathway_equations` and associated `transformer` for the
     * given `cell_type`. The transformer takes care of setting the respective
     * pathway inputs from the other problems (diffusion and optionally HCMM if
     * the pathway has a stress input). Note that it is assumed that all
     * pathways in the vector use the same transformer. Additionally, note that
     * the vector of pathways and the transformers are moved into the class.
     * Also, the vectors `pathway_equations` and `pathway_weights` must have the
     * same size. The `pathway_weights` vector can be used to determine the
     * importance of each pathway equation specified in `pathway_equations`.
     * Note that the sum of pathway weights must be equal to 1.
     *
     *  @param[in] cell_type CellType
     *  @param[in] pathway_equations a vector of (potentially different) pathway
     * equations associated with cell_type. Note that the all the pathways are
     * assumed to have the same size, i.e. the same number of nodes.
     * @param[in] pathway_weights_vector A vector of the same size as
     * pathway_equations containing the weights of each pathway equation
     * specified in pathway_equations. Sum of all the weights must be 1.
     *  @param[in] transformer
     *  @param[in] constituent_id of the associated constituent in the HCMM
     * problem, defaults to invalid number which means no constituent is
     * associated with the given pathway/no stress input.
     *  @param[in] material_id can be used to mark cells where this cell type is
     * present. Note that material_id is based on the dealii naming, it doesn't
     * have anything to do with the actual materials used in the MSHCMM.
     * Defaults to invalid_unsigned_int which means cell will be placed on every
     * element regardless of the element's material_id.
     *  @param[in] cell_localizer can be used to further refine the placement of
     * the cells if assignment by material_id is not sufficient. Default is that
     * there is no spatial restriction.
     *  //todo: remove cell_localizer? Necessary/helpful at all?
     */
    void
    add_pathways_and_transformers(
      const Cells::CellType &cell_type,
      std::vector<std::shared_ptr<Equations::PathwayEquationBase<Number>>>
                                           &&pathway_equations,
      const std::vector<Number>             &pathway_weights_vector,
      InputTransformerPathway<dim, Number> &&transformer,
      const unsigned int constituent_id = dealii::numbers::invalid_unsigned_int,
      const std::vector<unsigned int> &material_ids =
        {dealii::numbers::invalid_unsigned_int},
      CellLocalizer<dim> &&cell_localizer = [](const dealii::Point<dim> &p) {
        (void)p;
        return true;
      });

    //! Get entire pathway_storage.
    [[nodiscard]] const std::unordered_map<
      Cells::CellType,
      std::vector<std::shared_ptr<Equations::PathwayEquationBase<Number>>>> &
    get_pathway_storage() const;

    //! return the PathwayEquation associated with a given @p cell_type and @p pathway_id.
    const Equations::ODE<Number> &
    get_pathway_equation(const Cells::CellType &cell_type,
                         const unsigned int     pathway_id) const;

    //! return the weights of the pathways
    const std::vector<Number> &
    get_pathway_weights(const Cells::CellType &cell_type) const;

    /**
     * @brief Get the transformer pair for a specific `cell_type`.
     *
     * @param[in] cell_type
     *
     * @returns a TransformerPair. First element is the input transformer from diffusion values and
     * transferable parameters to the pathway, the second element is the
     * constituent_id from a constituent in the HCMM model.
     */
    [[nodiscard]] const TransformerPair<dim, Number> &
    get_transformer(const Cells::CellType &cell_type) const;

    //! Return the cell_localizer for the given cell_type
    [[nodiscard]] const CellLocalizer<dim> &
    get_cell_localizer(const Cells::CellType &cell_type) const;

    //! Return the material_id(s) where the given cell_type should be located
    [[nodiscard]] const std::vector<unsigned int> &
    get_cell_type_material_id(const Cells::CellType &cell_type) const;

    /**
     * @brief Add an endothelial cell layer to a boundary.
     *
     * @details
     * This function adds an endothelial cell layer to the pathway storage. It
     * is very similar to add_pathways_and_transformers() with the only
     * difference that you need to specify an `output_transformer` and provide a
     * `boundary_id`. The boundary_id specifies at which boundary the
     * endothelial cell layer is located.
     *
     * @pre Note that EndothelialCellLayer currently only supports 1
     * boundary_id, i.e., there cannot be more than one endothelial layer.
     *
     * @param cell_type
     * @param pathway_equations
     * @param endothelial_pathway_weights
     * @param input_transformer
     * @param output_transformer
     * @param boundary_id
     * @param cell_localizer
     */
    void
    add_endothelial_cell_layer(
      const Cells::CellType &cell_type,
      std::vector<std::shared_ptr<Equations::PathwayEquationBase<Number>>>
                               &&pathway_equations,
      const std::vector<Number> &endothelial_pathway_weights,
      InputTransformerEndothelialPathway<dim, Number>  &&input_transformer,
      OutputTransformerEndothelialPathway<dim, Number> &&output_transformer,
      const unsigned int                                 boundary_id,
      CellLocalizer<dim> &&cell_localizer = [](const dealii::Point<dim> &p) {
        (void)p;
        return true;
      });

    /**
     * @brief Get pointer to endothelial cell layer.
     *
     * @return SharedPtr to endothelial layer or nullptr if not present.
     */
    const std::shared_ptr<EndothelialCellLayer<dim, Number>>
    get_endothelial_cell_layer() const;

    /**
     * @brief Same as above but non-const.
     *
     * @return SharedPtr to endothelial layer or nullptr if not present.
     */
    std::shared_ptr<EndothelialCellLayer<dim, Number>>
    get_endothelial_cell_layer();

  private:
    // template<typename It>
    void
    print_insertion_status(/*It it,*/ bool success);

    std::unordered_map<
      Cells::CellType,
      std::vector<std::shared_ptr<Equations::PathwayEquationBase<Number>>>>
      pathway_storage;

    std::unordered_map<Cells::CellType, std::vector<Number>> pathway_weights;

    std::unordered_map<Cells::CellType, TransformerPair<dim, Number>>
      transformers;

    std::unordered_map<Cells::CellType, CellLocalizer<dim>> localizers;

    std::unordered_map<Cells::CellType, std::vector<unsigned int>>
      localizers_material_id;

    std::shared_ptr<EndothelialCellLayer<dim, Number>> endothelial_cell_layer{
      nullptr};
  };
} // namespace Pathways
