#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/hdf5.h>
#include <deal.II/base/quadrature_point_data.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/numerics/data_out.h>

#include <MSHCMM/common/fe_data.h>
#include <MSHCMM/pathways/cells/local_cell_collection.h>
#include <MSHCMM/pathways/pathway_storage.h>

#include <map>
#include <string>
#include <vector>

// todo: ADD documentation!!!


namespace Pathways
{

  template <int dim, typename Number = double>
  class PathwayOutput
  {
  public:
    explicit PathwayOutput(
      const Common::FEData<dim, Number> &fe_data_pathways,
      const bool                         use_single_cell_collection,
      const PathwayStorage<dim, Number> &pathway_storage,
      const dealii::CellDataStorage<
        typename dealii::Triangulation<dim>::cell_iterator,
        Cells::LocalCellCollection<dim, Number>> &local_cell_collections,
      const bool                                  write_vtu,
      const std::map<Cells::CellType, std::vector<unsigned int>> &cells_for_vtu,
      const bool         write_average    = false,
      const std::string &filename         = "pathway_output",
      const std::string &output_directory = "./",
      const MPI_Comm    &mpi_communicator = MPI_COMM_WORLD);


    void
    write_output(
      const unsigned int                 step,
      const Number                       time,
      const PathwayStorage<dim, Number> &pathway_storage,
      const dealii::CellDataStorage<
        typename dealii::Triangulation<dim>::cell_iterator,
        Cells::LocalCellCollection<dim, Number>> &local_cell_collections);

  private:
    const Common::FEData<dim, Number> &fe_data;
    const bool                         use_single_cell_collection;
    dealii::ConditionalOStream         pcout;

    bool write_vtu;
    bool write_average;

    // number of digits of dataset names
    // important that these are the same when file is created and when it is
    // opened during writing!
    const unsigned int output_digits{5};

    // name of the output file
    std::string filename;
    std::string output_dir;

    // VTU output stuff
    // map with cell types to include in vtu output
    std::map<Cells::CellType, std::vector<unsigned int>> cells_to_output;
    // map of outputs, note that each cell type might have several
    // subpopulations
    std::map<Cells::CellType,
             std::vector<dealii::LinearAlgebra::distributed::Vector<Number>>>
      pathway_outputs;
    // one dof_handler per cell type
    std::map<Cells::CellType, std::unique_ptr<dealii::DoFHandler<dim>>>
      pathway_dof_handlers;
    // component interpretation for each cell type and associated subpopulation
    std::map<
      Cells::CellType,
      std::vector<
        dealii::DataComponentInterpretation::DataComponentInterpretation>>
      output_component_interpretation;

    // node names for cell types and respective subpopulation
    std::map<Cells::CellType, std::vector<std::vector<std::string>>>
      output_names;

    // get max number of nodes of pathways in local cell collection
    unsigned int
    get_max_pathway_nodes(const std::vector<Cells::Cell<Number>> &cells) const;


    /**
     * @brief Setup all the necessary data structures for vtu output.
     *
     * @param pathway_storage
     * @param cells_for_vtu
     */
    void
    setup_vtu(const PathwayStorage<dim, Number> &pathway_storage,
              const std::map<Cells::CellType, std::vector<unsigned int>>
                &cells_for_vtu);

    /**
     * @brief Write a time step as vtu output.
     *
     * @param step
     * @param pathway_storage
     * @param local_cell_collections
     */
    void
    write_vtu_output(
      const unsigned int                 step,
      const Number                       time,
      const PathwayStorage<dim, Number> &pathway_storage,
      const dealii::CellDataStorage<
        typename dealii::Triangulation<dim>::cell_iterator,
        Cells::LocalCellCollection<dim, Number>> &local_cell_collections);
  };
} // namespace Pathways
