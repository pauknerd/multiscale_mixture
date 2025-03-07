#include <deal.II/grid/tria.h>

#include <MSHCMM/pathways/pathway_output.h>

#include <numeric>

#include "MSHCMM/utilities/helpers.h"

namespace Pathways
{

  template <int dim, typename Number>
  PathwayOutput<dim, Number>::PathwayOutput(
    const Common::FEData<dim, Number> &fe_data_pathways,
    const bool                         use_single_cell_collection,
    const PathwayStorage<dim, Number> &pathway_storage,
    const dealii::CellDataStorage<
      typename dealii::Triangulation<dim>::cell_iterator,
      Cells::LocalCellCollection<dim, Number>> &local_cell_collections,
    const bool                                  write_vtu,
    const std::map<Cells::CellType, std::vector<unsigned int>> &cells_for_vtu,
    const bool                                                  write_average,
    const std::string                                          &filename,
    const std::string &output_directory,
    const MPI_Comm    &mpi_communicator)
    : fe_data(fe_data_pathways)
    , use_single_cell_collection(use_single_cell_collection)
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , write_vtu(write_vtu)
    , write_average(write_average)
    , filename(filename)
    , output_dir(output_directory)
  {
    // setup necessary data structures
    if (write_vtu)
      setup_vtu(pathway_storage, cells_for_vtu);
  }


  template <int dim, typename Number>
  void
  PathwayOutput<dim, Number>::write_output(
    const unsigned int                 step,
    const Number                       time,
    const PathwayStorage<dim, Number> &pathway_storage,
    const dealii::CellDataStorage<
      typename dealii::Triangulation<dim>::cell_iterator,
      Cells::LocalCellCollection<dim, Number>> &local_cell_collections)
  {
    if (write_vtu)
      write_vtu_output(step, time, pathway_storage, local_cell_collections);
  }

  template <int dim, typename Number>
  unsigned int
  PathwayOutput<dim, Number>::get_max_pathway_nodes(
    const std::vector<Cells::Cell<Number>> &cells) const
  {
    // find cell with the largest cell_state
    const auto it = std::max_element(std::cbegin(cells),
                                     std::cend(cells),
                                     [](const auto &a, const auto &b) {
                                       return a.get_cell_state().size() <
                                              b.get_cell_state().size();
                                     });

    // return the cell state size of that cell
    return it->get_cell_state().size();
  }

  template <int dim, typename Number>
  void
  PathwayOutput<dim, Number>::setup_vtu(
    const PathwayStorage<dim, Number>                          &pathway_storage,
    const std::map<Cells::CellType, std::vector<unsigned int>> &cells_for_vtu)
  {
    Assert(
      !cells_for_vtu.empty(),
      dealii::ExcMessage(
        "write_vtu was set to true when setting up the pathway output but no cells were passed to include in vtu output. Either"
        " pass a map with cells to include in the output or set write_vtu to false."));

    // set which cells to print
    cells_to_output = cells_for_vtu;

    // need node names, can do that later, get from pathway storage
    const auto &pw_storage = pathway_storage.get_pathway_storage();

    // check if cells to output has some empty vectors, if yes, replace them
    // with all the index nodes for the given cell type
    for (auto &[cell_type, entries_to_print] : cells_to_output)
      {
        if (entries_to_print.empty())
          {
            // resize. Assumes all subpopulations have the same number of nodes!
            entries_to_print.resize(
              pw_storage.at(cell_type)[0]->n_components());
            // fill with values from 0 to n_components - 1
            std::iota(entries_to_print.begin(), entries_to_print.end(), 0);
          }
      }

    //// some setup work before looping over cells
    for (const auto &[cell_type, entries_to_print] : cells_to_output)
      {
        // need to create a dof handler for each cell type
        pathway_dof_handlers[cell_type] =
          std::make_unique<dealii::DoFHandler<dim>>(
            fe_data.get_triangulation());
        // number of pathway components to include
        const auto n_components = entries_to_print.size();

        // check that max entry in entries_to_print is smaller than size of that
        // pathway
        Assert(
          *(std::max_element(entries_to_print.cbegin(),
                             entries_to_print.cend())) <
            pw_storage.at(cell_type)[0]->n_components(),
          dealii::ExcMessage(
            "One or more of the indices you provided for pathway output for cell type " +
            Cells::CellType2string(cell_type) +
            " exceeds the number of nodes in the pathway equation."));

        // set output component interpretation
        output_component_interpretation[cell_type].resize(
          n_components,
          dealii::DataComponentInterpretation::component_is_scalar);

        // get the degree of the base element. This assumes that the
        // displacement FE is always first. In case of single cell collection
        // fe_degree is 1. (Only one cell collection in element center)
        const unsigned int fe_degree =
          use_single_cell_collection ? 1 : fe_data.get_fe().degree + 1;
        pathway_dof_handlers[cell_type]->distribute_dofs(dealii::FESystem<dim>(
          dealii::FE_DGQArbitraryNodes<dim>(dealii::QGauss<1>(fe_degree)),
          n_components));

        // get number of subpopulations for current cell type from pw_storage
        // todo: could be just one if we are only interested in AVERAGE pathway
        // output
        const auto n_subpops = pw_storage.at(cell_type).size();
        // reserve space for that many subpopulations
        pathway_outputs[cell_type].reserve(n_subpops);


        // add and resize vectors to hold the pathway output
        // add data vector, already resized
        for (size_t i = 0; i < n_subpops; ++i)
          {
            // add vector
            pathway_outputs[cell_type].emplace_back();
            // resize vector
            HELPERS::initialize_locally_owned_vector(
              pathway_outputs[cell_type].back(),
              *(pathway_dof_handlers.at(cell_type)));
          }

        // set nodes names
        for (unsigned int s = 0; s < n_subpops; ++s)
          {
            std::vector<std::string> node_names;
            for (const auto i : entries_to_print)
              {
                // since all subpopulations are assumed to have the same node
                // names
                node_names.push_back(
                  Cells::CellType2string(cell_type) + "_" +
                  dealii::Utilities::int_to_string(s) + "_" +
                  pw_storage.at(cell_type)[s]->get_node_names()[i]);
              }
            output_names[cell_type].push_back(node_names);
          }
      }
  }

  template <int dim, typename Number>
  void
  PathwayOutput<dim, Number>::write_vtu_output(
    const unsigned int                 step,
    const Number                       time,
    const PathwayStorage<dim, Number> &pathway_storage,
    const dealii::CellDataStorage<
      typename dealii::Triangulation<dim>::cell_iterator,
      Cells::LocalCellCollection<dim, Number>> &local_cell_collections)
  {
    dealii::DataOutBase::VtkFlags output_flags;
    output_flags.time                     = time;
    output_flags.write_higher_order_cells = true;
    output_flags.compression_level =
      dealii::DataOutBase::CompressionLevel::best_speed;

    dealii::DataOut<dim> data_out;
    data_out.set_flags(output_flags);

    // need node names, can do that later, get from pathway storage
    const auto &pw_storage = pathway_storage.get_pathway_storage();

    // check if cells are marked for output
    if (!cells_to_output.empty())
      {
        // need to reset global vectors in pathway outputs
        for (auto &[cell_type, vectors] : pathway_outputs)
          for (auto &v : vectors)
            v = 0.0;

        // need different quantities for each cell type
        std::map<Cells::CellType, unsigned int> dofs_per_cell_quad_points;
        std::map<Cells::CellType, std::vector<dealii::Vector<Number>>>
          element_rhs;
        std::map<Cells::CellType, std::vector<dealii::types::global_dof_index>>
          local_dof_indices;

        for (const auto &[cell_type, entries_to_print] : cells_to_output)
          {
            (void)entries_to_print;
            dofs_per_cell_quad_points[cell_type] =
              pathway_dof_handlers.at(cell_type)->get_fe().dofs_per_cell;
            // need number of subpopulations to initialize rhs vectors properly
            const unsigned int n_subpopulations =
              pw_storage.at(cell_type).size();
            element_rhs[cell_type] =
              std::vector(n_subpopulations,
                          dealii::Vector<Number>(
                            dofs_per_cell_quad_points.at(cell_type)));
            local_dof_indices[cell_type].resize(
              dofs_per_cell_quad_points.at(cell_type));
          }

        // original
        const auto &dof_handler = fe_data.get_dof_handler();
        // get iterators
        auto element = dof_handler.begin_active();
        // map of iterators
        std::map<Cells::CellType,
                 typename dealii::DoFHandler<dim>::active_cell_iterator>
          iterators;

        for (const auto &[cell_type, pw_dof_handler] : pathway_dof_handlers)
          {
            iterators[cell_type] = pw_dof_handler->begin_active();
          }

        for (; element != dof_handler.end();
             element++,
             std::for_each(iterators.begin(), iterators.end(), [](auto &it) {
               it.second++;
             }))
          {
            if (element->is_locally_owned())
              {
                // reset rhs for all cell types and subpopulations
                for (auto &[cell_type, vectors] : element_rhs)
                  for (auto &v : vectors)
                    v = 0;

                // get local cell collection on current element
                const auto local_cell_collection =
                  local_cell_collections.get_data(element);
                // get size of local_cell_collection, i.e. number of quadrature
                // points
                const unsigned int n_quad_points = local_cell_collection.size();
                // loop over quadrature points and assemble extracted quantities
                // into cell_rhs
                for (unsigned int q = 0; q < n_quad_points; ++q)
                  {
                    for (const auto &[cell_type, subpopulations] :
                         local_cell_collection[q]->get_cells())
                      {
                        // check if cell_type is in Cell_types_to_print
                        if (cells_to_output.find(cell_type) !=
                            cells_to_output.end())
                          {
                            // loop over subpopulations
                            for (unsigned int subpop = 0;
                                 subpop < subpopulations.size();
                                 ++subpop)
                              {
                                // get cell state of subpopulation
                                const auto &cell_state =
                                  subpopulations.at(subpop).get_cell_state();

                                // counter of index to write into
                                unsigned int counter = 0;

                                for (const auto i :
                                     cells_to_output.at(cell_type))
                                  {
                                    // add entry to rhs of current cell type and
                                    // subpopulation
                                    element_rhs.at(cell_type)[subpop](
                                      q + counter * n_quad_points) =
                                      cell_state[i];
                                    // increase counter
                                    ++counter;
                                  }
                              }
                          }
                      }
                  }

                // loop over all iterators and get local dof indices
                for (const auto &[cell_type, element_iterator] : iterators)
                  {
                    element_iterator->get_dof_indices(
                      local_dof_indices.at(cell_type));
                  }

                // assemble
                dealii::AffineConstraints<Number> affineConstraints;
                // need to loop over all cell types and associated
                // subpopulations loop over all cell types and subpopulations...
                for (const auto &[cell_type, subpopulations] : pw_storage)
                  {
                    if (cells_to_output.find(cell_type) !=
                        cells_to_output.end())
                      {
                        for (unsigned int subpop = 0;
                             subpop < subpopulations.size();
                             ++subpop)
                          {
                            affineConstraints.distribute_local_to_global(
                              element_rhs.at(cell_type)[subpop],
                              local_dof_indices.at(cell_type),
                              pathway_outputs.at(cell_type)[subpop]);
                          }
                      }
                  }
              }
          }
        // loop over all cell types and subpopulations...
        for (const auto &[cell_type, subpopulations] : pw_storage)
          {
            if (cells_to_output.find(cell_type) != cells_to_output.end())
              {
                for (unsigned int subpop = 0; subpop < subpopulations.size();
                     ++subpop)
                  {
                    // ...and sum up overlapping parts
                    pathway_outputs.at(cell_type)[subpop].compress(
                      dealii::VectorOperation::add);
                    // ...add data vector to data_out
                    data_out.add_data_vector(
                      *(pathway_dof_handlers.at(cell_type)),
                      pathway_outputs.at(cell_type)[subpop],
                      output_names.at(cell_type)[subpop],
                      output_component_interpretation.at(cell_type));
                  }
              }
          }
      }

    // build patches
    data_out.build_patches(fe_data.get_mapping(), fe_data.get_fe().degree);
    // write data
    data_out.write_vtu_with_pvtu_record(
      output_dir, filename, step, fe_data.get_MPI_comm(), 3);
  }

  // instantiations
  template class PathwayOutput<2, double>;
  template class PathwayOutput<3, double>;

} // namespace Pathways