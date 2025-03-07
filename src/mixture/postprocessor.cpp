#include <MSHCMM/mixture/growth_and_remodeling/postprocessor.h>
#include <MSHCMM/utilities/helpers.h>

namespace Mixture::Postprocessing
{

  template <int dim, typename Number>
  void
  Postprocessor<dim, Number>::write_output(
    const unsigned int                                step,
    const dealii::DoFHandler<dim>                    &dof_handler,
    const dealii::TrilinosWrappers::MPI::BlockVector &solution,
    const dealii::CellDataStorage<
      typename dealii::Triangulation<dim>::cell_iterator,
      LocalMixture<dim, Number>> &local_mixtures) const
  {
    dealii::DataOut<dim> data_out;
    std::vector<
      dealii::DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, dealii::DataComponentInterpretation::component_is_part_of_vector);
    std::vector<std::string> solution_name(dim, "displacement");
    // add pressure output
    solution_name.emplace_back("pressure");
    data_component_interpretation.push_back(
      dealii::DataComponentInterpretation::component_is_scalar);
    // add dilatation output
    solution_name.emplace_back("dilatation");
    data_component_interpretation.push_back(
      dealii::DataComponentInterpretation::component_is_scalar);

    dealii::DataOutBase::VtkFlags output_flags;
    output_flags.write_higher_order_cells = true;
    data_out.set_flags(output_flags);

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             solution_name,
                             dealii::DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    //// write subdomain id
    dealii::Vector<float> subdomain(
      dof_handler.get_triangulation().n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = dof_handler.get_triangulation().locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    //// write material id
    dealii::Vector<float> material_ids(
      dof_handler.get_triangulation().n_active_cells());
    unsigned int it = 0;
    for (const auto &element : dof_handler.active_cell_iterators())
      {
        material_ids(it) = element->material_id();
        ++it;
      }
    data_out.add_data_vector(material_ids, "material_id");

    //// write derived quantities for mixture and constituents
    // need all these quantities because they need to exist until the data is
    // written
    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>>
                            helper_vector_tensor_quantities;
    dealii::DoFHandler<dim> helper_dof_handler_tensor;

    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>>
                            scalar_quantities_mixture;
    dealii::DoFHandler<dim> mixture_dof_handler_scalar;

    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>>
                            scalar_constituent_quantities;
    dealii::DoFHandler<dim> constituents_dof_handler_scalar;

    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>>
                            vector_constituent_quantities;
    dealii::DoFHandler<dim> constituents_dof_handler_vector;

    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>>
                            tensor_constituent_quantities;
    dealii::DoFHandler<dim> constituents_dof_handler_tensor;

    write_derived_quantities(dof_handler,
                             local_mixtures,
                             data_out,
                             helper_vector_tensor_quantities,
                             helper_dof_handler_tensor,
                             scalar_quantities_mixture,
                             mixture_dof_handler_scalar,
                             scalar_constituent_quantities,
                             constituents_dof_handler_scalar,
                             vector_constituent_quantities,
                             constituents_dof_handler_vector,
                             tensor_constituent_quantities,
                             constituents_dof_handler_tensor);

    // todo: get mapping?
    data_out.build_patches(dealii::MappingQ<dim>(
                             dof_handler.get_fe().base_element(0).degree),
                           dof_handler.get_fe().base_element(0).degree);

    // write vtu
    data_out.write_vtu_with_pvtu_record(
      output_directory_, filename_, step, dof_handler.get_communicator(), 3);
  }

  template <int dim, typename Number>
  void
  Postprocessor<dim, Number>::write_output(
    const unsigned int                           step,
    const dealii::DoFHandler<dim>               &dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector &solution,
    const dealii::CellDataStorage<
      typename dealii::Triangulation<dim>::cell_iterator,
      LocalMixture<dim, Number>> &local_mixtures) const
  {
    dealii::DataOut<dim> data_out;
    std::vector<
      dealii::DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, dealii::DataComponentInterpretation::component_is_part_of_vector);
    std::vector<std::string> solution_name(dim, "displacement");
    // output settings
    dealii::DataOutBase::VtkFlags output_flags;
    output_flags.write_higher_order_cells = true;
    data_out.set_flags(output_flags);

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             solution_name,
                             dealii::DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    //// write subdomain id
    dealii::Vector<float> subdomain(
      dof_handler.get_triangulation().n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = dof_handler.get_triangulation().locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    //// write material id
    dealii::Vector<float> material_ids(
      dof_handler.get_triangulation().n_active_cells());
    unsigned int it = 0;
    for (const auto &element : dof_handler.active_cell_iterators())
      {
        material_ids(it) = element->material_id();
        ++it;
      }
    data_out.add_data_vector(material_ids, "material_id");

    //// write derived quantities for mixture and constituents
    // need all these quantities because they need to exist until the data is
    // written
    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>>
                            helper_vector_tensor_quantities;
    dealii::DoFHandler<dim> helper_dof_handler_tensor;

    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>>
                            scalar_quantities_mixture;
    dealii::DoFHandler<dim> mixture_dof_handler_scalar;

    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>>
                            scalar_constituent_quantities;
    dealii::DoFHandler<dim> constituents_dof_handler_scalar;

    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>>
                            vector_constituent_quantities;
    dealii::DoFHandler<dim> constituents_dof_handler_vector;

    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>>
                            tensor_constituent_quantities;
    dealii::DoFHandler<dim> constituents_dof_handler_tensor;

    write_derived_quantities(dof_handler,
                             local_mixtures,
                             data_out,
                             helper_vector_tensor_quantities,
                             helper_dof_handler_tensor,
                             scalar_quantities_mixture,
                             mixture_dof_handler_scalar,
                             scalar_constituent_quantities,
                             constituents_dof_handler_scalar,
                             vector_constituent_quantities,
                             constituents_dof_handler_vector,
                             tensor_constituent_quantities,
                             constituents_dof_handler_tensor);

    // todo: get mapping?
    data_out.build_patches(dealii::MappingQ<dim>(
                             dof_handler.get_fe().base_element(0).degree),
                           dof_handler.get_fe().base_element(0).degree);

    data_out.write_vtu_with_pvtu_record(
      output_directory_, filename_, step, dof_handler.get_communicator(), 3);
  }


  template <int dim, typename Number>
  void
  Postprocessor<dim, Number>::write_derived_quantities(
    const dealii::DoFHandler<dim> &dof_handler,
    const dealii::CellDataStorage<
      typename dealii::Triangulation<dim>::cell_iterator,
      LocalMixture<dim, Number>> &local_mixtures,
    dealii::DataOut<dim>         &data_out,
    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>>
                            &helper_vector_tensor_quantities,
    dealii::DoFHandler<dim> &helper_dof_handler_tensor,
    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>>
                            &scalar_quantities_mixture,
    dealii::DoFHandler<dim> &mixture_dof_handler_scalar,
    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>>
                            &scalar_constituent_quantities,
    dealii::DoFHandler<dim> &constituents_dof_handler_scalar,
    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>>
                            &vector_constituent_quantities,
    dealii::DoFHandler<dim> &constituents_dof_handler_vector,
    std::vector<dealii::LinearAlgebra::distributed::Vector<Number>>
                            &tensor_constituent_quantities,
    dealii::DoFHandler<dim> &constituents_dof_handler_tensor) const
  {
    //// Scalar quantities on mixture level
    if (!scalar_quantity_extractors.empty())
      {
        // setup helper_dof_handler_tensor
        mixture_dof_handler_scalar.reinit(dof_handler.get_triangulation());
        // number of total components of a scalar quantity
        const auto n_components = 1;
        // get the degree of the base element. This assumes that the
        // displacement FE is always first
        mixture_dof_handler_scalar.distribute_dofs(dealii::FESystem<dim>(
          dealii::FE_DGQArbitraryNodes<dim>(
            dealii::QGauss<1>(dof_handler.get_fe().base_element(0).degree + 1)),
          n_components));

        // NOTE: this call is ESSENTIAL for the function to work. Otherwise,
        // dealii throws an error related to WorkStream, might be related to
        // add_data_vector() and how it tries to do stuff in parallel
        scalar_quantities_mixture.reserve(scalar_quantity_extractors.size());

        // loop over all tensor quantity extractors
        for (const auto &[quantity_name, quantity_extractor] :
             scalar_quantity_extractors)
          {
            // add data vector and resize using the helper dof handler
            scalar_quantities_mixture.emplace_back();
            HELPERS::initialize_locally_owned_vector(
              scalar_quantities_mixture.back(), mixture_dof_handler_scalar);

            // pass data vector that was last added to the function
            const auto [output_names, output_component_interpretation] =
              write_scalar_quantity(quantity_name,
                                    dof_handler,
                                    local_mixtures,
                                    quantity_extractor,
                                    mixture_dof_handler_scalar,
                                    scalar_quantities_mixture.back());

            // add data vector to data_out
            data_out.add_data_vector(mixture_dof_handler_scalar,
                                     scalar_quantities_mixture.back(),
                                     output_names,
                                     output_component_interpretation);
          }
      }


    //// Tensor quantities on mixture level
    if (!tensor_quantity_extractors.empty())
      {
        // setup helper_dof_handler_tensor
        helper_dof_handler_tensor.reinit(dof_handler.get_triangulation());
        // number of total components of a tensor in dim dimensions = dim^2
        const auto n_components = (unsigned int)std::pow(dim, 2);
        // get the degree of the base element. This assumes that the
        // displacement FE is always first
        helper_dof_handler_tensor.distribute_dofs(dealii::FESystem<dim>(
          dealii::FE_DGQArbitraryNodes<dim>(
            dealii::QGauss<1>(dof_handler.get_fe().base_element(0).degree + 1)),
          n_components));

        // NOTE: this call us ESSENTIAL for the function to work. Otherwise,
        // dealii throws an error related to WorkStream, might be related to
        // add_data_vector() and how it tries to do stuff in parallel
        helper_vector_tensor_quantities.reserve(
          tensor_quantity_extractors.size());

        // loop over all tensor quantity extractors
        for (const auto &[quantity_name, quantity_extractor] :
             tensor_quantity_extractors)
          {
            // add data vector and resize using the helper dof handler
            helper_vector_tensor_quantities.emplace_back();
            HELPERS::initialize_locally_owned_vector(
              helper_vector_tensor_quantities.back(),
              helper_dof_handler_tensor);

            // pass data vector that was last added to the function
            const auto [output_names, output_component_interpretation] =
              write_tensor_quantity(quantity_name,
                                    dof_handler,
                                    local_mixtures,
                                    quantity_extractor,
                                    helper_dof_handler_tensor,
                                    helper_vector_tensor_quantities.back());

            // add data vector to data_out
            data_out.add_data_vector(helper_dof_handler_tensor,
                                     helper_vector_tensor_quantities.back(),
                                     output_names,
                                     output_component_interpretation);
          }
      }


    //// Scalar quantities on constituent level
    if (!scalar_quantity_extractors_constituents.empty())
      {
        // setup helper_dof_handler_tensor
        constituents_dof_handler_scalar.reinit(dof_handler.get_triangulation());
        // number of total components of a scalar is 1
        const unsigned int n_components = 1;
        constituents_dof_handler_scalar.distribute_dofs(dealii::FESystem<dim>(
          dealii::FE_DGQArbitraryNodes<dim>(
            dealii::QGauss<1>(dof_handler.get_fe().base_element(0).degree + 1)),
          n_components));

        // NOTE: this call us ESSENTIAL for the function to work. Otherwise,
        // dealii throws an error related to WorkStream, might be related to
        // add_data_vector() and how it tries to do stuff in parallel
        scalar_constituent_quantities.reserve(
          scalar_quantity_extractors_constituents.size());

        // loop over all tensor quantity extractors
        for (const auto &output_params :
             scalar_quantity_extractors_constituents)
          {
            const auto &[name, id, quantity_extractor] = output_params;
            // add data vector, already resized
            scalar_constituent_quantities.emplace_back();
            HELPERS::initialize_locally_owned_vector(
              scalar_constituent_quantities.back(),
              constituents_dof_handler_scalar);
            // pass data vector that was last added to the function
            const auto [output_names, output_component_interpretation] =
              write_scalar_quantity_constituent(
                name,
                id,
                dof_handler,
                local_mixtures,
                quantity_extractor,
                constituents_dof_handler_scalar,
                scalar_constituent_quantities.back());

            // add data vector to data_out
            data_out.add_data_vector(constituents_dof_handler_scalar,
                                     scalar_constituent_quantities.back(),
                                     output_names,
                                     output_component_interpretation);
          }
      }


    //// Vector quantities on constituent level
    if (!vector_quantity_extractors_constituents.empty())
      {
        // setup helper_dof_handler_tensor
        constituents_dof_handler_vector.reinit(dof_handler.get_triangulation());
        // number of total components of a vector in dim dimensions = dim
        const auto n_components = dim;
        constituents_dof_handler_vector.distribute_dofs(dealii::FESystem<dim>(
          dealii::FE_DGQArbitraryNodes<dim>(
            dealii::QGauss<1>(dof_handler.get_fe().base_element(0).degree + 1)),
          n_components));

        // NOTE: this call us ESSENTIAL for the function to work. Otherwise,
        // dealii throws an error related to WorkStream, might be related to
        // add_data_vector() and how it tries to do stuff in parallel
        vector_constituent_quantities.reserve(
          vector_quantity_extractors_constituents.size());

        // loop over all vector quantity extractors
        for (const auto &output_params :
             vector_quantity_extractors_constituents)
          {
            const auto &[name, id, quantity_extractor] = output_params;
            // add data vector, already resized
            vector_constituent_quantities.emplace_back();
            HELPERS::initialize_locally_owned_vector(
              vector_constituent_quantities.back(),
              constituents_dof_handler_vector);
            // pass data vector that was last added to the function
            const auto [output_names, output_component_interpretation] =
              write_vector_quantity_constituent(
                name,
                id,
                dof_handler,
                local_mixtures,
                quantity_extractor,
                constituents_dof_handler_vector,
                vector_constituent_quantities.back());

            // add data vector to data_out
            data_out.add_data_vector(constituents_dof_handler_vector,
                                     vector_constituent_quantities.back(),
                                     output_names,
                                     output_component_interpretation);
          }
      }


    //// Tensor quantities on constituent level
    if (!tensor_quantity_extractors_constituents.empty())
      {
        // setup helper_dof_handler_tensor
        constituents_dof_handler_tensor.reinit(dof_handler.get_triangulation());
        // number of total components of a tensor in dim dimensions = dim^2
        const auto n_components = (unsigned int)std::pow(dim, 2);
        constituents_dof_handler_tensor.distribute_dofs(dealii::FESystem<dim>(
          dealii::FE_DGQArbitraryNodes<dim>(
            dealii::QGauss<1>(dof_handler.get_fe().base_element(0).degree + 1)),
          n_components));

        // NOTE: this call is ESSENTIAL for the function to work. Otherwise,
        // dealii throws an error related to WorkStream, might be related to
        // add_data_vector() and how it tries to do stuff in parallel
        tensor_constituent_quantities.reserve(
          tensor_quantity_extractors_constituents.size());

        // loop over all tensor quantity extractors
        for (const auto &output_params :
             tensor_quantity_extractors_constituents)
          {
            const auto &[name, id, quantity_extractor] = output_params;
            // add data vector, already resized
            tensor_constituent_quantities.emplace_back();
            HELPERS::initialize_locally_owned_vector(
              tensor_constituent_quantities.back(),
              constituents_dof_handler_tensor);
            // pass data vector that was last added to the function
            const auto [output_names, output_component_interpretation] =
              write_tensor_quantity_constituent(
                name,
                id,
                dof_handler,
                local_mixtures,
                quantity_extractor,
                constituents_dof_handler_tensor,
                tensor_constituent_quantities.back());

            // add data vector to data_out
            data_out.add_data_vector(constituents_dof_handler_tensor,
                                     tensor_constituent_quantities.back(),
                                     output_names,
                                     output_component_interpretation);
          }
      }
  }

  // todo: add checks for out of bounds?
  template <int dim, typename Number>
  void
  Postprocessor<dim, Number>::add_quantity_extractor_scalar(
    const std::string                    &quantity_name,
    const QuantityExtractor<dim, Number> &quantity_extractor_scalar)
  {
    scalar_quantity_extractors.insert(
      {quantity_name, quantity_extractor_scalar});
  }

  template <int dim, typename Number>
  void
  Postprocessor<dim, Number>::add_quantity_extractor_vector(
    const std::string                    &quantity_name,
    const QuantityExtractor<dim, Number> &quantity_extractor_vector)
  {
    vector_quantity_extractors.insert(
      {quantity_name, quantity_extractor_vector});
  }

  template <int dim, typename Number>
  void
  Postprocessor<dim, Number>::add_quantity_extractor_tensor(
    const std::string                    &quantity_name,
    const QuantityExtractor<dim, Number> &quantity_extractor_tensor)
  {
    tensor_quantity_extractors.insert(
      {quantity_name, quantity_extractor_tensor});
  }

  template <int dim, typename Number>
  void
  Postprocessor<dim, Number>::add_quantity_extractor_scalar_constituent(
    const std::map<std::string, unsigned int>       &name_and_id,
    const QuantityExtractorConstituent<dim, Number> &quantity_extractor_scalar)
  {
    for (const auto &[name, id] : name_and_id)
      scalar_quantity_extractors_constituents.emplace_back(
        std::make_tuple(name, id, quantity_extractor_scalar));
  }

  template <int dim, typename Number>
  void
  Postprocessor<dim, Number>::add_quantity_extractor_vector_constituent(
    const std::map<std::string, unsigned int>       &name_and_id,
    const QuantityExtractorConstituent<dim, Number> &quantity_extractor_vector)
  {
    for (const auto &[name, id] : name_and_id)
      vector_quantity_extractors_constituents.emplace_back(
        std::make_tuple(name, id, quantity_extractor_vector));
  }

  template <int dim, typename Number>
  void
  Postprocessor<dim, Number>::add_quantity_extractor_tensor_constituent(
    const std::map<std::string, unsigned int>       &name_and_id,
    const QuantityExtractorConstituent<dim, Number> &quantity_extractor_tensor)
  {
    for (const auto &[name, id] : name_and_id)
      tensor_quantity_extractors_constituents.emplace_back(
        std::make_tuple(name, id, quantity_extractor_tensor));
  }

  // output for a tensor quantity
  template <int dim, typename Number>
  std::pair<std::vector<std::string>,
            std::vector<
              dealii::DataComponentInterpretation::DataComponentInterpretation>>
  Postprocessor<dim, Number>::write_tensor_quantity(
    const std::string             &quantity_name,
    const dealii::DoFHandler<dim> &dof_handler,
    const dealii::CellDataStorage<
      typename dealii::Triangulation<dim>::cell_iterator,
      LocalMixture<dim, Number>>                       &local_mixtures,
    const QuantityExtractor<dim, Number>               &extractor,
    const dealii::DoFHandler<dim>                      &helper_dof_handler,
    dealii::LinearAlgebra::distributed::Vector<Number> &helper_vector) const
  {
    // number of total components of a tensor in dim dimensions = dim^2
    const auto n_components = (unsigned int)std::pow(dim, 2);

    // delegate to base function
    write_output_base(dof_handler,
                      local_mixtures,
                      extractor,
                      helper_dof_handler,
                      helper_vector);

    // set names
    const std::vector<std::string> quantity_names(n_components, quantity_name);

    // set component interpretation
    const std::vector<
      dealii::DataComponentInterpretation::DataComponentInterpretation>
      component_interpretation(
        n_components,
        dealii::DataComponentInterpretation::component_is_part_of_tensor);

    return std::make_pair(quantity_names, component_interpretation);
  }

  template <int dim, typename Number>
  std::pair<std::vector<std::string>,
            std::vector<
              dealii::DataComponentInterpretation::DataComponentInterpretation>>
  Postprocessor<dim, Number>::write_scalar_quantity(
    const std::string             &quantity_name,
    const dealii::DoFHandler<dim> &dof_handler,
    const dealii::CellDataStorage<
      typename dealii::Triangulation<dim>::cell_iterator,
      LocalMixture<dim, Number>>                       &local_mixtures,
    const QuantityExtractor<dim, Number>               &extractor,
    const dealii::DoFHandler<dim>                      &helper_dof_handler,
    dealii::LinearAlgebra::distributed::Vector<Number> &helper_vector) const
  {
    // number of total components of a scalar is 1
    const auto n_components = 1;

    // delegate to base function
    write_output_base(dof_handler,
                      local_mixtures,
                      extractor,
                      helper_dof_handler,
                      helper_vector);

    // set names
    const std::vector<std::string> quantity_names(n_components, quantity_name);

    // set component interpretation
    const std::vector<
      dealii::DataComponentInterpretation::DataComponentInterpretation>
      component_interpretation(
        n_components, dealii::DataComponentInterpretation::component_is_scalar);

    return std::make_pair(quantity_names, component_interpretation);
  }

  // write output for a scalar quantity on constituent level
  template <int dim, typename Number>
  std::pair<std::vector<std::string>,
            std::vector<
              dealii::DataComponentInterpretation::DataComponentInterpretation>>
  Postprocessor<dim, Number>::write_scalar_quantity_constituent(
    const std::string             &quantity_name,
    const unsigned int             constituent_id,
    const dealii::DoFHandler<dim> &dof_handler,
    const dealii::CellDataStorage<
      typename dealii::Triangulation<dim>::cell_iterator,
      LocalMixture<dim, Number>>                       &local_mixtures,
    const QuantityExtractorConstituent<dim, Number>    &extractor,
    const dealii::DoFHandler<dim>                      &helper_dof_handler,
    dealii::LinearAlgebra::distributed::Vector<Number> &helper_vector) const
  {
    // number of total components of a scalar is 1
    const unsigned int n_components = 1;

    // delegate to base function
    write_output_base_constituent(constituent_id,
                                  dof_handler,
                                  local_mixtures,
                                  extractor,
                                  helper_dof_handler,
                                  helper_vector);

    // set names
    const std::vector<std::string> quantity_names(n_components, quantity_name);

    // set component interpretation
    const std::vector<
      dealii::DataComponentInterpretation::DataComponentInterpretation>
      component_interpretation(
        n_components, dealii::DataComponentInterpretation::component_is_scalar);

    return std::make_pair(quantity_names, component_interpretation);
  }

  // write output for a vector quantity on constituent level
  template <int dim, typename Number>
  std::pair<std::vector<std::string>,
            std::vector<
              dealii::DataComponentInterpretation::DataComponentInterpretation>>
  Postprocessor<dim, Number>::write_vector_quantity_constituent(
    const std::string             &quantity_name,
    const unsigned int             constituent_id,
    const dealii::DoFHandler<dim> &dof_handler,
    const dealii::CellDataStorage<
      typename dealii::Triangulation<dim>::cell_iterator,
      LocalMixture<dim, Number>>                       &local_mixtures,
    const QuantityExtractorConstituent<dim, Number>    &extractor,
    const dealii::DoFHandler<dim>                      &helper_dof_handler,
    dealii::LinearAlgebra::distributed::Vector<Number> &helper_vector) const
  {
    // number of total components of a vector in dim dimensions is dim
    const unsigned int n_components = dim;

    // delegate to base function
    write_output_base_constituent(constituent_id,
                                  dof_handler,
                                  local_mixtures,
                                  extractor,
                                  helper_dof_handler,
                                  helper_vector);

    // set names
    const std::vector<std::string> quantity_names(n_components, quantity_name);

    // set component interpretation
    const std::vector<
      dealii::DataComponentInterpretation::DataComponentInterpretation>
      component_interpretation(
        n_components,
        dealii::DataComponentInterpretation::component_is_part_of_vector);

    return std::make_pair(quantity_names, component_interpretation);
  }

  // write output for a tensor quantity on constituent level
  template <int dim, typename Number>
  std::pair<std::vector<std::string>,
            std::vector<
              dealii::DataComponentInterpretation::DataComponentInterpretation>>
  Postprocessor<dim, Number>::write_tensor_quantity_constituent(
    const std::string             &quantity_name,
    const unsigned int             constituent_id,
    const dealii::DoFHandler<dim> &dof_handler,
    const dealii::CellDataStorage<
      typename dealii::Triangulation<dim>::cell_iterator,
      LocalMixture<dim, Number>>                       &local_mixtures,
    const QuantityExtractorConstituent<dim, Number>    &extractor,
    const dealii::DoFHandler<dim>                      &helper_dof_handler,
    dealii::LinearAlgebra::distributed::Vector<Number> &helper_vector) const
  {
    // number of total components of a tensor in dim dimensions = dim^2
    const auto n_components = (unsigned int)std::pow(dim, 2);

    // delegate to base function
    write_output_base_constituent(constituent_id,
                                  dof_handler,
                                  local_mixtures,
                                  extractor,
                                  helper_dof_handler,
                                  helper_vector);

    // set names
    const std::vector<std::string> quantity_names(n_components, quantity_name);

    // set component interpretation
    const std::vector<
      dealii::DataComponentInterpretation::DataComponentInterpretation>
      component_interpretation(
        n_components,
        dealii::DataComponentInterpretation::component_is_part_of_tensor);

    return std::make_pair(quantity_names, component_interpretation);
  }

  template <int dim, typename Number>
  void
  Postprocessor<dim, Number>::write_output_base(
    const dealii::DoFHandler<dim> &dof_handler,
    const dealii::CellDataStorage<
      typename dealii::Triangulation<dim>::cell_iterator,
      LocalMixture<dim, Number>>                       &local_mixtures,
    const QuantityExtractor<dim, Number>               &extractor,
    const dealii::DoFHandler<dim>                      &helper_dof_handler,
    dealii::LinearAlgebra::distributed::Vector<Number> &helper_vector) const
  {
    const unsigned int dofs_per_cell_quad_points =
      helper_dof_handler.get_fe().dofs_per_cell;
    dealii::Vector<Number> cell_rhs(dofs_per_cell_quad_points);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell_quad_points);

    auto cell       = dof_handler.begin_active();
    auto cell_quads = helper_dof_handler.begin_active();

    for (; cell != dof_handler.end(); cell++, cell_quads++)
      {
        if (cell->is_locally_owned())
          {
            cell_rhs = 0;

            // get LocalMixtures on current cell
            const auto local_mixture = local_mixtures.get_data(cell);
            // apply passed extractor function
            extractor(local_mixture, cell_rhs);
            // get dof indices
            cell_quads->get_dof_indices(local_dof_indices);

            // assemble
            dealii::AffineConstraints<Number> affineConstraints;
            affineConstraints.distribute_local_to_global(cell_rhs,
                                                         local_dof_indices,
                                                         helper_vector);
          }
      }
    // sum up overlapping parts
    helper_vector.compress(dealii::VectorOperation::add);
  }

  template <int dim, typename Number>
  void
  Postprocessor<dim, Number>::write_output_base_constituent(
    const unsigned int             constituent_id,
    const dealii::DoFHandler<dim> &dof_handler,
    const dealii::CellDataStorage<
      typename dealii::Triangulation<dim>::cell_iterator,
      LocalMixture<dim, Number>>                       &local_mixtures,
    const QuantityExtractorConstituent<dim, Number>    &extractor,
    const dealii::DoFHandler<dim>                      &helper_dof_handler,
    dealii::LinearAlgebra::distributed::Vector<Number> &helper_vector) const
  {
    const unsigned int dofs_per_cell_quad_points =
      helper_dof_handler.get_fe().dofs_per_cell;
    dealii::Vector<Number> cell_rhs(dofs_per_cell_quad_points);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell_quad_points);

    auto cell       = dof_handler.begin_active();
    auto cell_quads = helper_dof_handler.begin_active();

    for (; cell != dof_handler.end(); cell++, cell_quads++)
      {
        if (cell->is_locally_owned())
          {
            cell_rhs = 0;

            // get LocalMixtures on current cell
            const auto local_mixture = local_mixtures.get_data(cell);
            // apply passed extractor function
            extractor(local_mixture, constituent_id, cell_rhs);
            // get dof indices
            cell_quads->get_dof_indices(local_dof_indices);

            // assemble
            dealii::AffineConstraints<Number> affineConstraints;
            affineConstraints.distribute_local_to_global(cell_rhs,
                                                         local_dof_indices,
                                                         helper_vector);
          }
      }
    // sum up overlapping parts
    helper_vector.compress(dealii::VectorOperation::add);
  }

  template <int dim, typename Number>
  void
  Postprocessor<dim, Number>::set_output_file_name(
    const std::string &filename,
    const std::string &output_directory)
  {
    filename_         = filename;
    output_directory_ = output_directory;
  }

  // instantiations
  template class Postprocessor<3, double>;
} // namespace Mixture::Postprocessing
