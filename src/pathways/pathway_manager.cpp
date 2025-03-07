#include <MSHCMM/pathways/pathway_manager.h>

namespace Pathways
{
  template <int dim, typename Number>
  PathwayManager<dim, Number>::PathwayManager(
    const typename dealii::SUNDIALS::ARKode<
      dealii::Vector<Number>>::AdditionalData &data,
    const MPI_Comm                            &mpi_communicator)
    : mpi_communicator(mpi_communicator)
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(pcout,
                      dealii::TimerOutput::never,
                      dealii::TimerOutput::wall_times)
    , pathway_solver_data(data)
    , ODE_solver(
        std::make_unique<dealii::SUNDIALS::ARKode<dealii::Vector<Number>>>(
          data,
          mpi_communicator))
  {}

  template <int dim, typename Number>
  void
  PathwayManager<dim, Number>::setup(
    PathwayStorage<dim, Number> &&pathway_storage_in,
    Common::FEData<dim, Number> &&fe_data_in,
    const bool                    use_single_cell_collection)
  {
    use_single_cell_collection_ = use_single_cell_collection;
    this->fe_data               = std::move(fe_data_in);
    pathway_storage             = std::move(pathway_storage_in);
  }

  template <int dim, typename Number>
  void
  PathwayManager<dim, Number>::setup(
    const PathwayStorage<dim, Number> &pathway_storage_in,
    const dealii::Triangulation<dim>  &triangulation,
    const unsigned int                 fe_degree,
    const unsigned int                 quad_degree,
    const bool                         use_single_cell_collection)
  {
    use_single_cell_collection_ = use_single_cell_collection;
    this->fe_data.reinit(triangulation, fe_degree, quad_degree, 1);
    pathway_storage = (pathway_storage_in);
  }

  template <int dim, typename Number>
  void
  PathwayManager<dim, Number>::distribute_cells()
  {
    dealii::TimerOutput::Scope timer_section(computing_timer,
                                             "Pathways - Distribute cells");

    pcout << "Setting up cell data with spatial distribution..." << std::endl;
    // check that pathway_storage is not empty
    Assert(
      !(pathway_storage.is_empty()),
      dealii::ExcMessage(
        "Cannot distribute cells if pathway storage is empty, add pathways first!"));

    // if single_cell_collection == true, create fe_values object with midpoint
    // quadrature rule (i.e. just a single quadrature point in the cell center).
    // Created via lambda
    auto fe_values =
      [](const Common::FEData<dim, Number> &fe_data_,
         const bool single_cell_collection) -> dealii::FEValues<dim> {
      if (single_cell_collection)
        return fe_data_.make_fe_values(dealii::QMidpoint<dim>(),
                                       dealii::update_quadrature_points);
      else
        return fe_data_.make_fe_values(dealii::update_quadrature_points);
    }(this->fe_data, use_single_cell_collection_);

    // get number of cell collections, i.e., number of quadrature points on an
    // element
    const auto n_cells_per_element = fe_values.get_quadrature().size();

    // create filter to extract locally owned cells
    using CellFilter = dealii::FilteredIterator<
      typename dealii::parallel::TriangulationBase<dim>::active_cell_iterator>;
    // initialize CellDataStorage
    local_cell_collections.initialize(
      CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
                 fe_data.get_triangulation().begin_active()),
      CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
                 fe_data.get_triangulation().end()),
      n_cells_per_element);

    // loop over all elements
    for (const auto &element :
         fe_data.get_triangulation().active_cell_iterators())
      {
        if (element->is_locally_owned())
          {
            // get so far empty local data on the element
            const std::vector<
              std::shared_ptr<Cells::LocalCellCollection<dim, Number>>>
              local_cell_collection = local_cell_collections.get_data(element);

            // reinit fe_values
            fe_values.reinit(element);
            // get quadrature points
            const auto &quadrature_points = fe_values.get_quadrature_points();

            // loop over number of quadrature points
            for (unsigned int q_point = 0; q_point < n_cells_per_element;
                 ++q_point)
              {
                Assert(
                  quadrature_points.size() == n_cells_per_element,
                  dealii::ExcMessage(
                    "Size of quadrature_points and n_cells_per_element does not match! "
                    "Only 1 or n_quad_points_diffusion cells can be distributed!"));

                // current quadrature point coordinates
                const auto &current_q_point_coords = quadrature_points[q_point];
                // set location of local cell collection
                local_cell_collection[q_point]->set_location(
                  current_q_point_coords);

                // loop over pathway_storage and create cells
                for (const auto &[cell_type, vector_of_pathways] :
                     pathway_storage.get_pathway_storage())
                  {
                    // check if cell_type is present on current element with
                    // given material_id. If not skip this cell_type and move to
                    // the next
                    const auto &mat_ids =
                      pathway_storage.get_cell_type_material_id(cell_type);

                    if (mat_ids.front() ==
                          dealii::numbers::invalid_unsigned_int or
                        std::find(mat_ids.cbegin(),
                                  mat_ids.cend(),
                                  element->material_id()) != mat_ids.cend())
                      {
                        // number of pathways for cell_type specific pathway
                        // const unsigned int n_pathways =
                        //  vector_of_pathways.size();

                        // get weights of pathways for current cell_type
                        const auto pathway_weights =
                          pathway_storage.get_pathway_weights(cell_type);

                        // get number of components of this pathway (assumed to
                        // be the same for all pathway equations of a given
                        // cell_type) so we just use the first one since we have
                        // at least one cell
                        const unsigned int n_components_pathway =
                          vector_of_pathways[0]->n_components();
                        const unsigned int n_input_components =
                          vector_of_pathways[0]->n_inputs();
                        const unsigned int n_output_components =
                          vector_of_pathways[0]->n_outputs();

                        // get cell localizer for cell type
                        const auto &cell_localizer =
                          pathway_storage.get_cell_localizer(cell_type);
                        // add cell to local cell collection depending on
                        // specified cell localizer in PathwayStorage
                        if (cell_localizer(current_q_point_coords))
                          {
                            // todo: what if we want to add several cells of the
                            // same cell type
                            //  to the same local cell collection?
                            // the number of cells added is determined by the
                            // size of the pathway_weights vector passed
                            local_cell_collection[q_point]->add_cells(
                              cell_type,
                              pathway_weights,
                              n_components_pathway,
                              n_input_components,
                              n_output_components);
                          }
                      }
                  }
              }
          }
      }
    const unsigned int distributed_cells =
      use_single_cell_collection_ ?
        fe_data.get_triangulation().n_global_active_cells() :
        fe_data.get_triangulation().n_global_active_cells() *
          fe_data.get_quadrature().size();

    pcout << "Distributed " << distributed_cells << " cells on "
          << fe_data.get_triangulation().n_global_active_cells() << " elements!"
          << std::endl;

    // distribute endothelial cell layer if pathway storage has one
    if (pathway_storage.get_endothelial_cell_layer())
      distribute_endothelial_cells();

    pcout << "Setting up cell data...DONE!" << std::endl;
  }

  template <int dim, typename Number>
  void
  PathwayManager<dim, Number>::distribute_endothelial_cells()
  {
    pcout << "Setting up endothelial cell data with spatial distribution..."
          << std::endl;
    // check that pathway_storage is not empty
    Assert(
      !(pathway_storage.is_empty()),
      dealii::ExcMessage(
        "Cannot distribute cells if pathway storage is empty, add pathways first!"));

    // if single_cell_collection == true, create fe_values object with midpoint
    // quadrature rule (i.e. just a single quadrature point in the cell center).
    // Created via lambda
    auto fe_face_values =
      [](const Common::FEData<dim, Number> &fe_data_,
         const bool single_cell_collection) -> dealii::FEFaceValues<dim> {
      if (single_cell_collection)
        return fe_data_.make_fe_face_values(dealii::QMidpoint<dim - 1>(),
                                            dealii::update_quadrature_points);
      else
        return fe_data_.make_fe_face_values(dealii::update_quadrature_points);
    }(this->fe_data, use_single_cell_collection_);

    // get number of cell collections, i.e., number of quadrature points on an
    // element
    const auto n_cells_per_element = fe_face_values.get_quadrature().size();

    // create filter to extract locally owned cells
    using CellFilter = dealii::FilteredIterator<
      typename dealii::parallel::TriangulationBase<dim>::active_cell_iterator>;

    // initialize CellDataStorage
    const auto boundary_id = get_endothelium_boundary_id();
    // create filtered iterators
    // note: {} around boundary_id important to avoid call of operator()
    dealii::FilteredIterator<
      typename dealii::parallel::TriangulationBase<dim>::active_cell_iterator>
      cell(BoundaryIdEqualTo({boundary_id})),
      endc(BoundaryIdEqualTo({boundary_id}), fe_data.get_triangulation().end());
    cell.set_to_next_positive(fe_data.get_triangulation().begin_active());

    // initialize each cell individually, based on dealii test
    // (https://github.com/dealii/dealii/blob/master/tests/base/cell_data_storage_01.cc)
    for (; cell != endc; ++cell)
      {
        // std::cout << "positive!" << std::endl;
        local_endothelial_cell_collections.initialize(cell,
                                                      n_cells_per_element);

        // loop over faces
        for (const auto &face : cell->face_iterators())
          {
            if (!face->at_boundary())
              continue;
            if (face->boundary_id() == boundary_id)
              {
                // get so far empty local data on the element
                const std::vector<
                  std::shared_ptr<Cells::LocalCellCollection<dim, Number>>>
                  local_endothelial_cell_collection =
                    local_endothelial_cell_collections.get_data(cell);

                // reinit fe_values
                fe_face_values.reinit(cell, face);
                // get quadrature points
                const auto &quadrature_points =
                  fe_face_values.get_quadrature_points();

                // loop over number of quadrature points
                for (unsigned int q_point = 0; q_point < n_cells_per_element;
                     ++q_point)
                  {
                    Assert(
                      quadrature_points.size() == n_cells_per_element,
                      dealii::ExcMessage(
                        "Size of quadrature_points and n_cells_per_element does not match! "
                        "Only 1 or n_quad_points_diffusion cells can be distributed!"));

                    // current quadrature point coordinates
                    const auto &current_q_point_coords =
                      quadrature_points[q_point];
                    // set location of local cell collection
                    local_endothelial_cell_collection[q_point]->set_location(
                      current_q_point_coords);

                    // loop over pathway_storage and create cells
                    const auto &cell_type =
                      pathway_storage.get_endothelial_cell_layer()
                        ->get_endothelial_cell_type();
                    const auto &vector_of_pathways =
                      pathway_storage.get_endothelial_cell_layer()
                        ->get_endothelial_pathways();

                    // get pathway weights for current cell type
                    const auto pathway_weights =
                      pathway_storage.get_pathway_weights(cell_type);

                    // get number of components of this pathway (assumed to be
                    // the same for all pathway equations of a given cell_type)
                    // so we just use the first one since we have at least one
                    // cell
                    const unsigned int n_components_pathway =
                      vector_of_pathways[0]->n_components();
                    const unsigned int n_input_components =
                      vector_of_pathways[0]->n_inputs();
                    const unsigned int n_output_components =
                      vector_of_pathways[0]->n_outputs();

                    // get cell localizer for cell type
                    const auto &cell_localizer =
                      pathway_storage.get_cell_localizer(cell_type);
                    // add cell to local cell collection depending on specified
                    // cell localizer in PathwayStorage
                    if (cell_localizer(current_q_point_coords))
                      {
                        // todo: what if we want to add several cells of the
                        // same cell type
                        //  to the same local cell collection?
                        // the number of cells added is determined by the size
                        // of the pathway_ids vector passed
                        local_endothelial_cell_collection[q_point]->add_cells(
                          cell_type,
                          pathway_weights,
                          n_components_pathway,
                          n_input_components,
                          n_output_components);
                      }
                  }
              }
          }

        // DEBUG TEST
        // loop over all elements
        //        for (const auto& element :
        //        fe_data.get_triangulation().active_cell_iterators())
        //          {
        //            const auto data =
        //            local_endothelial_cell_collections.template
        //            try_get_data(element);
        //
        //            if (data)
        //              std::cout << "size of data: " << data->size() <<
        //              std::endl;
        //          }
      }
    pcout << "Setting up endothelial cell data...DONE!" << std::endl;
  }

  template <int dim, typename Number>
  void
  PathwayManager<dim, Number>::solve_pathways_on_element(
    const std::vector<dealii::Point<dim>> &p,
    const Number                           time,
    const std::vector<std::shared_ptr<Cells::LocalCellCollection<dim, Number>>>
                                              &local_cell_collection,
    const std::vector<dealii::Vector<Number>> &diffusion_values,
    const std::vector<std::shared_ptr<Mixture::LocalMixture<dim, Number>>>
        &local_mixture,
    bool store_baseline)
  {
    dealii::TimerOutput::Scope timer_section(computing_timer,
                                             "Pathways - Solve on element");

    // todo: add check if local_cell_collection is empty? Could theoretically be
    // the case depending on how the cells are localized, e.g., no cells if x <
    // 0.5

    // number of cell_collections on element
    const unsigned int n_cell_collections_per_element =
      local_cell_collection.size();
    // in case we only have 1 cell collection, the values of the diffusion
    // problem on the element will be averaged
    [[maybe_unused]] dealii::Vector<Number> avg_diffusion_values;
    // here we need a vector because we have several constituents
    [[maybe_unused]] std::vector<
      Mixture::Constituents::TransferableParameters<Number>>
      avg_transferable_parameters;

    // if only one cell collection on the element, average diffusion_values and
    // transferable_parameters of the quadrature points
    if (n_cell_collections_per_element == 1)
      {
        compute_average_of_quadrature_vectors(diffusion_values,
                                              avg_diffusion_values);
        if (!local_mixture.empty())
          compute_average_transferable_parameters(local_mixture,
                                                  avg_transferable_parameters);
      }

    // loop over number of cell collections on the element, could be less than
    // number of quadrature points used in the diffusion problem!
    for (unsigned int q_point = 0; q_point < n_cell_collections_per_element;
         ++q_point)
      {
        // get local cell collection at quadrature point
        auto &cell_collection_qp = local_cell_collection[q_point];
        // get diffusion values at quadrature point
        const auto &diffusion_values_qp = diffusion_values[q_point];

        // solve pathways of cells at current quadrature point
        // loop over cells in cell_collection_qp
        for (auto &[cell_type, cells] : cell_collection_qp->get_cells())
          {
            // get InputTransformerTuple for cell type
            const auto &[transformer, constituent_id] =
              pathway_storage.get_transformer(cell_type);

            // loop over all cells of that cell type
            for (auto &cell : cells)
              {
                // get pathway_id
                const unsigned int pathway_id = cell.get_pathway_id();
                // get pathway input
                auto &cell_state = cell.get_cell_state();
                // get diffusion values
                const auto &diff_values = n_cell_collections_per_element == 1 ?
                                            avg_diffusion_values :
                                            diffusion_values_qp;
                // empty transferable parameters
                Mixture::Constituents::TransferableParameters<Number>
                  transferable_parameters;
                // check if constituent_id is valid, if yes, get transferable
                // parameters of this constituent
                if (constituent_id != dealii::numbers::invalid_unsigned_int)
                  {
                    // DEBUG
                    // pcout << "Transferring fiber stress to pathway..." <<
                    // std::endl;

                    // if only one cell per element, average the fiber stresses
                    // of that constituent
                    transferable_parameters =
                      n_cell_collections_per_element == 1 ?
                        avg_transferable_parameters[constituent_id] :
                        local_mixture[q_point]->get_transferable_parameters(
                          constituent_id);
                  }

                // get mixture stress
                dealii::Tensor<2, dim, Number>          deformation_gradient;
                dealii::SymmetricTensor<2, dim, Number> PK2_stress;
                // todo: adjust for single cell collection!
                if (!local_mixture.empty())
                  {
                    deformation_gradient =
                      local_mixture[q_point]
                        ->get_mixture_deformation_gradient();
                    PK2_stress = local_mixture[q_point]->get_mixture_stress();
                  }

                // apply transformer: get stress of constituent and add it
                // to the pathway equation based on the specified transformer
                transformer(p[q_point],
                            time,
                            diff_values,
                            transferable_parameters,
                            pathway_id,
                            deformation_gradient,
                            PK2_stress,
                            cell_state);

                // get pathway equation based on cell_type and pathway_id
                ODE_solver->explicit_function =
                  pathway_storage.get_pathway_equation(cell.get_cell_type(),
                                                       cell.get_pathway_id());
                // solve equation
                ODE_solver->solve_ode(cell.get_cell_state());
                // DEBUG
                // print cell state for debugging
                // pcout << "cell state after solving: " << std::endl;
                // pcout << cell.get_cell_state() << std::endl;

                // this is only called if store_baseline is set to true which
                // should be done during equilibration
                if (store_baseline)
                  cell.store_baseline_pathway_output();
              }
          }
      }
  }

  template <int dim, typename Number>
  void
  PathwayManager<dim, Number>::solve_pathways_on_boundary(
    const std::vector<dealii::Point<dim>> &p,
    const Number                           time,
    const std::vector<std::shared_ptr<Cells::LocalCellCollection<dim, Number>>>
      &local_endothelial_cell_collection,
    const std::vector<dealii::Vector<Number>> &diffusion_values,
    const std::vector<dealii::Vector<Number>> &displacements,
    bool                                       store_baseline)
  {
    dealii::TimerOutput::Scope timer_section(computing_timer,
                                             "Pathways - Solve on boundary");

    // number of cell_collections on element
    const unsigned int n_cell_collections_per_element =
      local_endothelial_cell_collection.size();
    // in case we only have 1 cell collection, the values of the diffusion
    // problem and the displacements on the element will be averaged
    [[maybe_unused]] dealii::Vector<Number> avg_diffusion_values;
    [[maybe_unused]] dealii::Vector<Number> avg_displacements;

    // if only one cell collection on the element, average diffusion_values of
    // the quadrature points
    if (n_cell_collections_per_element == 1)
      {
        compute_average_of_quadrature_vectors(diffusion_values,
                                              avg_diffusion_values);
        // compute average displacements if they are not empty
        if (!displacements.empty())
          compute_average_of_quadrature_vectors(displacements,
                                                avg_displacements);
      }

    // loop over number of cell collections on the element, could be less than
    // number of quadrature points used in the diffusion problem!
    for (unsigned int q_point = 0; q_point < n_cell_collections_per_element;
         ++q_point)
      {
        // get local cell collection at quadrature point
        auto &cell_collection_qp = local_endothelial_cell_collection[q_point];
        // get diffusion values at quadrature point
        const auto &diffusion_values_qp = n_cell_collections_per_element == 1 ?
                                            avg_diffusion_values :
                                            diffusion_values[q_point];
        // get displacement values at quadrature point
        const auto &displacements_qp = n_cell_collections_per_element == 1 ?
                                         avg_displacements :
                                         displacements[q_point];

        // solve pathways of cells at current quadrature point
        // loop over cells in cell_collection_qp
        for (auto &[cell_type, cells] : cell_collection_qp->get_cells())
          {
            // get InputTransformer for endothelial cells
            const auto &transformer =
              pathway_storage.get_endothelial_cell_layer()
                ->get_input_transformer();

            // loop over all cells of that cell type
            for (auto &cell : cells)
              {
                // todo: not needed?
                //                // get pathway_id
                //                const unsigned int pathway_id =
                //                cell.get_pathway_id();
                // get pathway input
                auto &cell_state = cell.get_cell_state();

                // apply transformer
                transformer(p[q_point],
                            time,
                            displacements_qp,
                            diffusion_values_qp,
                            cell_state);

                // get pathway equation based on cell_type and pathway_id
                ODE_solver->explicit_function =
                  pathway_storage.get_endothelial_cell_layer()
                    ->get_pathway_equation(cell.get_pathway_id());
                // solve equation
                ODE_solver->solve_ode(cell_state);
                // DEBUG
                // print cell state for debugging
                // pcout << "endothelial cell state after solving: " <<
                // std::endl; pcout << cell_state << std::endl;

                // this is only called if store_baseline is set to true which
                // should be done during equilibration
                if (store_baseline)
                  cell.store_baseline_pathway_output();
              }
          }
      }
  }

  // todo: why not return a reference? probably because that is what
  // get_data(element) returns...
  template <int dim, typename Number>
  std::vector<std::shared_ptr<Cells::LocalCellCollection<dim, Number>>>
  PathwayManager<dim, Number>::get_local_cell_collection(
    const CellIteratorType<dim> &element)
  {
    return local_cell_collections.get_data(element);
  }

  template <int dim, typename Number>
  std::vector<std::shared_ptr<Cells::LocalCellCollection<dim, Number>>>
  PathwayManager<dim, Number>::get_local_endothelial_cell_collection(
    const CellIteratorType<dim> &element)
  {
    return local_endothelial_cell_collections.get_data(element);
  }

  // get quadrature point data
  template <int dim, typename Number>
  const dealii::CellDataStorage<CellIteratorType<dim>,
                                Cells::LocalCellCollection<dim, Number>> &
  PathwayManager<dim, Number>::get_cell_data_storage() const
  {
    return local_cell_collections;
  }

  // set final time
  template <int dim, typename Number>
  void
  PathwayManager<dim, Number>::reset_ODE_solver(const Number final_time)
  {
    // modify pathway_solver_data with new final time
    pathway_solver_data.initial_time = 0.0;
    pathway_solver_data.final_time   = final_time;
    // create new solver
    ODE_solver.reset(new dealii::SUNDIALS::ARKode<dealii::Vector<Number>>(
      pathway_solver_data, fe_data.get_MPI_comm()));
  }

  // compute average diffusion value
  template <int dim, typename Number>
  void
  PathwayManager<dim, Number>::compute_average_of_quadrature_vectors(
    const std::vector<dealii::Vector<Number>> &qp_data,
    dealii::Vector<Number>                    &avg_qp_data) const
  {
    // set size of avg_qp_data
    avg_qp_data.reinit(qp_data[0]);
    // add all vectors in qp_data
    std::for_each(qp_data.cbegin(),
                  qp_data.cend(),
                  [&avg_qp_data](const dealii::Vector<Number> &val) {
                    avg_qp_data += val;
                  });
    // divide by number of quadrature points which is equal to the size of
    // qp_data
    avg_qp_data /= qp_data.size();
  }

  template <int dim, typename Number>
  void
  PathwayManager<dim, Number>::compute_average_transferable_parameters(
    const std::vector<std::shared_ptr<Mixture::LocalMixture<dim, Number>>>
      &local_mixtures,
    std::vector<Mixture::Constituents::TransferableParameters<Number>>
      &avg_transferable_parameters) const
  {
    // set proper size
    avg_transferable_parameters.resize(local_mixtures[0]->n_constituents());
    // add all transferable parameters
    std::for_each(
      local_mixtures.cbegin(),
      local_mixtures.cend(),
      [&avg_transferable_parameters](
        const std::shared_ptr<Mixture::LocalMixture<dim, Number>> &lm) {
        for (size_t constituent_id = 0; constituent_id < lm->n_constituents();
             ++constituent_id)
          {
            // check that all the local mixtures at the quadrature points have
            // the same number of constituents
            Assert(
              avg_transferable_parameters.size() == lm->n_constituents(),
              dealii::ExcMessage(
                "Cannot compute average transferable parameters! Same number of constituents "
                "at all quadrature points is required!"));
            avg_transferable_parameters[constituent_id] +=
              lm->get_transferable_parameters(constituent_id);
          }
      });
    // divide by number of quadrature points in the problem which is equal to
    // the size of local_mixtures
    for (auto &avg_tp : avg_transferable_parameters)
      avg_tp /= local_mixtures.size();
  }

  template <int dim, typename Number>
  void
  PathwayManager<dim, Number>::setup_pathway_output(
    const std::string                                          &filename,
    const bool                                                  write_vtu,
    const std::map<Cells::CellType, std::vector<unsigned int>> &cells_for_vtu,
    const std::string &output_directory,
    const bool         write_average)
  {
    // create pathway output
    pathway_output =
      std::make_unique<PathwayOutput<dim, Number>>(fe_data,
                                                   use_single_cell_collection_,
                                                   pathway_storage,
                                                   local_cell_collections,
                                                   write_vtu,
                                                   cells_for_vtu,
                                                   write_average,
                                                   filename,
                                                   output_directory,
                                                   mpi_communicator);
  }

  template <int dim, typename Number>
  void
  PathwayManager<dim, Number>::write_output(const unsigned int step,
                                            const Number       time)
  {
    // only write pathway output if it has been set up
    if (pathway_output)
      {
        dealii::TimerOutput::Scope timer_section(computing_timer,
                                                 "Pathways - Output results");

        pathway_output->write_output(step,
                                     time,
                                     pathway_storage,
                                     local_cell_collections);
      }
  }

  template <int dim, typename Number>
  bool
  PathwayManager<dim, Number>::uses_single_cell_collection() const
  {
    return use_single_cell_collection_;
  }

  template <int dim, typename Number>
  void
  PathwayManager<dim, Number>::print_timer_stats(
    const bool print_mpi_stats) const
  {
    pcout << "Pathway Timings";
    // always print summary
    computing_timer.print_summary();
    // optionally, print mpi stats
    if (print_mpi_stats)
      computing_timer.print_wall_time_statistics(mpi_communicator);
  }

  template class PathwayManager<2, double>;
  template class PathwayManager<3, double>;
} // namespace Pathways
