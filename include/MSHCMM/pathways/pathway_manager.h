#pragma once

#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/timer.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/sundials/arkode.h>

#include <MSHCMM/common/fe_data.h>
#include <MSHCMM/diffusion/local_diffusion_parameters.h>
#include <MSHCMM/mixture/growth_and_remodeling/local_mixture.h>
#include <MSHCMM/pathways/cells/local_cell_collection.h>
#include <MSHCMM/pathways/pathway_output.h>
#include <MSHCMM/pathways/pathway_storage.h>

#include <iostream>
#include <random>
#include <vector>

namespace Pathways
{
  /**
   * @class PathwayManager
   * @brief Manages everything related to pathways.
   *
   *
   * @tparam dim
   * @tparam Number
   */
  template <int dim, typename Number = double>
  class PathwayManager
  {
  public:
    // todo: add setup functions to constructor?
    explicit PathwayManager(const typename dealii::SUNDIALS::ARKode<
                              dealii::Vector<Number>>::AdditionalData &data =
                              typename dealii::SUNDIALS::ARKode<
                                dealii::Vector<Number>>::AdditionalData(),
                            const MPI_Comm &mpi_communicator = MPI_COMM_WORLD);

    /**
     * One of the two setup functions MUST be called.
     *
     * @param pathway_storage_in
     * @param fe_data_in
     * @param use_single_cell_collection
     */
    void
    setup(PathwayStorage<dim, Number> &&pathway_storage_in,
          Common::FEData<dim, Number> &&fe_data_in,
          const bool                    use_single_cell_collection = false);

    /**
     * One of the two setup functions MUST be called.
     *
     * @param pathway_storage_in
     * @param triangulation
     * @param fe_degree
     * @param quad_degree
     * @param use_single_cell_collection
     */
    void
    setup(const PathwayStorage<dim, Number> &pathway_storage_in,
          const dealii::Triangulation<dim>  &triangulation,
          const unsigned int                 fe_degree,
          const unsigned int                 quad_degree,
          const bool use_single_cell_collection = false);

    /**
     * @brief Initialize data structures for pathway output.
     *
     *
     * @note VTU output always outputs the entire domain but you have the
     * option of selecting certain nodes that you want to output. If no node
     * selection is given, all the nodes are printed. It is however required to
     * at lease specify the cell types you want to include in the output.
     *
     * @param filename
     * @param write_vtu
     * @param cells_for_vtu cell type and a vector of indices of the nodes in the pathway equation to output.
     * @param write_average write average cell output of there is more than one pathway equation for a cell type.
     */
    void
    setup_pathway_output(
      const std::string &filename,
      const bool         write_vtu,
      const std::map<Cells::CellType, std::vector<unsigned int>>
                        &cells_for_vtu    = {},
      const std::string &output_directory = "./",
      const bool         write_average    = false);

    void
    write_output(const unsigned int step, const Number time);

    /** @brief Distribute cells to the LocalCellCollections.
     *
     * This function creates LocalCellCollections based on the stored
     * PathwayStorage. All the knowledge about which cells to include in the
     * LocalCellCollection are stored in the PathwayStorage. Pathways can be
     * assigned randomly.
     */
    void
    distribute_cells();
    // todo: not sure if that makes sense though. Cells could also just be
    // distributed via initial condition

    void
    distribute_endothelial_cells();
    // todo: not sure if that makes sense though. Cells could also just be
    // distributed via initial condition



    /**
     * @brief Solves all the pathways that are associated to the cells in the given cell collection
     * based on the current diffusion values and local mixture (optional).
     *
     * @details
     * The function loops over all cells in the given cell collections, applies
     * the stored transformers to assign the correct inputs to the cell state
     * (i.e., the initial condition of the pathway equation) for the current
     * time step, and solve this equation. The inputs to the pathways come from
     * the current solution of the diffusion problem `diffusion_values` and
     * optionally from the current state of the homogenized constrained mixture
     * model `local_mixture` (defaults to an empty vector). Additionally, this
     * function can optionally store the baseline parameters of the pathways
     * which can be useful to equilibrate the pathways, i.e., to set a baseline
     * state. The spatial coordinates and the time can be used to mock values.
     *
     * @param[in] p spatial coordinates of the quadrature points (i.e. local
     * cell collections, note that it could be just a single one in case
     * single_cell_collection is set to true).
     * @param[in] time current simulation time
     * @param[in] local_cell_collection
     * @param[in] diffusion_values
     * @param[in] local_mixture
     * @param store_baseline
     */
    void
    solve_pathways_on_element(
      const std::vector<dealii::Point<dim>> &p,
      const Number                           time,
      const std::vector<
        std::shared_ptr<Cells::LocalCellCollection<dim, Number>>>
                                                &local_cell_collection,
      const std::vector<dealii::Vector<Number>> &diffusion_values,
      const std::vector<std::shared_ptr<Mixture::LocalMixture<dim, Number>>>
        &local_mixture =
          std::vector<std::shared_ptr<Mixture::LocalMixture<dim, Number>>>{},
      bool store_baseline = false);

    /**
     * @brief Solve endothelial pathways on element boundary.
     *
     * @details
     * Solves the endothelial pathways based on the current diffusion values.
     * The spatial coordinates and the time can be used to mock values by
     * capturing dealii::Functions in the associated transformer. Note that this
     * could also be used to prescribe some external substances that are
     * transported in the vessel and absorbed by the endothelial layer.
     *
     * @param[in] p spatial coordinates of the quadrature points (i.e. local
     * cell collections, note that it could be just a single one in case
     * single_cell_collection is set to true).
     * @param[in] time current simulation time
     * @param[in] local_cell_collection
     * @param[in] diffusion_values
     * @param[in] displacements displacements at the given quadrature point(s).
     * Can be used to compute the current radius which can in turn be used to
     * approximate the current wall shear stress as is done in classical
     * constrained mixture models. The wall shear stress can then serve as an
     * input to the pathway.
     * @param store_baseline Flag indicating if the baseline values should be stored, usually only needed
     * when equilibrating the pathways.
     */
    void
    solve_pathways_on_boundary(
      const std::vector<dealii::Point<dim>> &p,
      const Number                           time,
      const std::vector<
        std::shared_ptr<Cells::LocalCellCollection<dim, Number>>>
        &local_endothelial_cell_collection,
      const std::vector<dealii::Vector<Number>> &diffusion_values,
      const std::vector<dealii::Vector<Number>> &displacements,
      bool                                       store_baseline = false);

    //! get local cell collection
    std::vector<std::shared_ptr<Cells::LocalCellCollection<dim, Number>>>
    get_local_cell_collection(const CellIteratorType<dim> &element);

    //! get local endothelial cell collection
    std::vector<std::shared_ptr<Cells::LocalCellCollection<dim, Number>>>
    get_local_endothelial_cell_collection(const CellIteratorType<dim> &element);

    //! get quadrature point data
    const dealii::CellDataStorage<CellIteratorType<dim>,
                                  Cells::LocalCellCollection<dim, Number>> &
    get_cell_data_storage() const;

    /**
     * @brief Set the final time in the ODE solver. Note that the final time can be considered as
     * the time step size of the pathway problem.
     *
     * @param final_time
     */
    void
    reset_ODE_solver(const Number final_time);

    [[nodiscard]] bool
    uses_single_cell_collection() const;

    [[nodiscard]] unsigned int
    get_endothelium_boundary_id() const
    {
      return pathway_storage.get_endothelial_cell_layer() ?
               pathway_storage.get_endothelial_cell_layer()
                 ->get_endothelium_boundary_id() :
               dealii::numbers::invalid_unsigned_int;
    }

    [[nodiscard]] bool
    has_endothelial_cells() const
    {
      return pathway_storage.get_endothelial_cell_layer() ? true : false;
    }

    std::shared_ptr<EndothelialCellLayer<dim, Number>>
    get_endothelial_cell_layer() const
    {
      return pathway_storage.get_endothelial_cell_layer();
    }

    [[nodiscard]] const dealii::TimerOutput &
    get_timer() const
    {
      return computing_timer;
    }

    void
    print_timer_stats(const bool print_mpi_stats = false) const;

  private:
    /**
     * @brief Compute average of given vector of dealii::Vectors of all quadrature points.
     *
     * @param[in] qp_data Vectors to average
     * @param[in,out] avg_qp_data Result
     */
    void
    compute_average_of_quadrature_vectors(
      const std::vector<dealii::Vector<Number>> &qp_data,
      dealii::Vector<Number>                    &avg_qp_data) const;

    /**
     * @brief Compute average transferable parameters on an element. Assumes same number of constituents
     * at all quadrature points!
     * */
    void
    compute_average_transferable_parameters(
      const std::vector<std::shared_ptr<Mixture::LocalMixture<dim, Number>>>
        &local_mixtures,
      std::vector<Mixture::Constituents::TransferableParameters<Number>>
        &avg_transferable_parameters) const;

    MPI_Comm                   mpi_communicator;
    dealii::ConditionalOStream pcout;
    dealii::TimerOutput        computing_timer;

    // LocalCellCollections at quadrature points
    dealii::CellDataStorage<CellIteratorType<dim>,
                            Cells::LocalCellCollection<dim, Number>>
      local_cell_collections;

    // Endothelial Cells - LocalCellCollections at quadrature points
    dealii::CellDataStorage<CellIteratorType<dim>,
                            Cells::LocalCellCollection<dim, Number>>
      local_endothelial_cell_collections;

    // vector holding the pathways that are in the system
    PathwayStorage<dim, Number>                 pathway_storage;
    std::unique_ptr<PathwayOutput<dim, Number>> pathway_output;

    // only needed for setup purposes, should be initialized with same fe_degree
    // as diffusion and HCMM
    Common::FEData<dim, Number> fe_data;
    bool                        use_single_cell_collection_{false};

    // ARKode solver (or AMICI)
    typename dealii::SUNDIALS::ARKode<dealii::Vector<Number>>::AdditionalData
      pathway_solver_data;
    std::unique_ptr<dealii::SUNDIALS::ARKode<dealii::Vector<Number>>>
      ODE_solver;
  };
} // namespace Pathways
