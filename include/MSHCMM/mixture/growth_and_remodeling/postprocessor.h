#pragma once

#include <deal.II/base/quadrature_point_data.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_out.h>

#include <MSHCMM/common/fe_data.h>

#include <fstream>
#include <functional>
#include <map>
#include <tuple>
#include <utility>

#include "local_mixture.h"
#include "postprocessing_functions.h"

namespace Mixture::Postprocessing
{

  template <int dim, typename Number = double>
  class NewPostprocessor : public dealii::DataPostprocessor<dim>
  {
  public:
    NewPostprocessor(const dealii::CellDataStorage<
                     typename dealii::Triangulation<dim>::cell_iterator,
                     LocalMixture<dim, Number>> &local_mixtures)
      : local_mixtures(local_mixtures) {};

    void
    evaluate_vector_field(
      const dealii::DataPostprocessorInputs::Vector<dim> &inputs,
      std::vector<dealii::Vector<double>> &computed_quantities) const override
    {
      const unsigned int n_quadrature_points = inputs.solution_values.size();
      Assert(inputs.solution_gradients.size() == n_quadrature_points,
             dealii::ExcInternalError());
      Assert(computed_quantities.size() == n_quadrature_points,
             dealii::ExcInternalError());

      for (unsigned int q = 0; q < n_quadrature_points; ++q)
        {
          // extract displacement gradient
          dealii::Tensor<2, dim> grad_u;
          for (unsigned int d = 0; d < dim; ++d)
            grad_u[d] = inputs.solution_gradients[q][d];

          // assemble deformation gradient into vector
          for (unsigned int d = 0; d < dim; ++d)
            for (unsigned int e = 0; e < dim; ++e)
              computed_quantities
                [q][dealii::Tensor<2, dim>::component_to_unrolled_index(
                  dealii::TableIndices<2>(d, e))] =
                  inputs.solution_gradients[q][d][e];
        }
    };

    std::vector<std::string>
    get_names() const override
    {
      std::vector<std::string> solution_names(dim * dim, "disp_grad");

      return solution_names;
    };

    std::vector<
      dealii::DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override
    {
      std::vector<
        dealii::DataComponentInterpretation::DataComponentInterpretation>
        interpretation(
          dim * dim,
          dealii::DataComponentInterpretation::component_is_part_of_tensor);

      return interpretation;
    };

    dealii::UpdateFlags
    get_needed_update_flags() const override
    {
      return dealii::update_values | dealii::update_gradients |
             dealii::update_quadrature_points;
    };

  private:
    const dealii::CellDataStorage<
      typename dealii::Triangulation<dim>::cell_iterator,
      LocalMixture<dim, Number>> &local_mixtures;
  };

  template <int dim, typename Number = double>
  class Postprocessor
  {
  public:
    void
    set_output_file_name(const std::string &filename,
                         const std::string &output_directory = "./");

    void
    write_output(const unsigned int                                step,
                 const dealii::DoFHandler<dim>                    &dof_handler,
                 const dealii::TrilinosWrappers::MPI::BlockVector &solution,
                 const dealii::CellDataStorage<
                   typename dealii::Triangulation<dim>::cell_iterator,
                   LocalMixture<dim, Number>> &local_mixtures) const;

    void
    write_output(const unsigned int                           step,
                 const dealii::DoFHandler<dim>               &dof_handler,
                 const dealii::TrilinosWrappers::MPI::Vector &solution,
                 const dealii::CellDataStorage<
                   typename dealii::Triangulation<dim>::cell_iterator,
                   LocalMixture<dim, Number>> &local_mixtures) const;

    //    void
    //    write_output(const unsigned int step,
    //                 const dealii::DoFHandler<dim>& dof_handler, const
    //                 dealii::PETScWrappers::MPI::Vector&         solution,
    //                 const dealii::CellDataStorage<typename
    //                 dealii::Triangulation<dim>::cell_iterator,
    //                                               LocalMixture<dim, Number>>&
    //                                               local_mixtures) const;
    // todo: add checks for out of bounds?
    void
    add_quantity_extractor_scalar(
      const std::string                    &quantity_name,
      const QuantityExtractor<dim, Number> &quantity_extractor_scalar);

    void
    add_quantity_extractor_vector(
      const std::string                    &quantity_name,
      const QuantityExtractor<dim, Number> &quantity_extractor_vector);

    void
    add_quantity_extractor_tensor(
      const std::string                    &quantity_name,
      const QuantityExtractor<dim, Number> &quantity_extractor_tensor);

    void
    add_quantity_extractor_scalar_constituent(
      const std::map<std::string, unsigned int> &name_and_id,
      const QuantityExtractorConstituent<dim, Number>
        &quantity_extractor_scalar);

    void
    add_quantity_extractor_vector_constituent(
      const std::map<std::string, unsigned int> &name_and_id,
      const QuantityExtractorConstituent<dim, Number>
        &quantity_extractor_vector);

    void
    add_quantity_extractor_tensor_constituent(
      const std::map<std::string, unsigned int> &name_and_id,
      const QuantityExtractorConstituent<dim, Number>
        &quantity_extractor_tensor);

  private:
    void
    write_derived_quantities(
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
      dealii::DoFHandler<dim> &constituents_dof_handler_tensor) const;

    // filename
    std::string filename_{"output"};
    std::string output_directory_{"./"};

    // quantity extractors on mixture level
    std::map<std::string, QuantityExtractor<dim, Number>>
      scalar_quantity_extractors;
    std::map<std::string, QuantityExtractor<dim, Number>>
      vector_quantity_extractors;
    std::map<std::string, QuantityExtractor<dim, Number>>
      tensor_quantity_extractors;

    // quantity extractors on constituent level
    std::vector<std::tuple<std::string,
                           unsigned int,
                           QuantityExtractorConstituent<dim, Number>>>
      scalar_quantity_extractors_constituents;
    std::vector<std::tuple<std::string,
                           unsigned int,
                           QuantityExtractorConstituent<dim, Number>>>
      vector_quantity_extractors_constituents;
    std::vector<std::tuple<std::string,
                           unsigned int,
                           QuantityExtractorConstituent<dim, Number>>>
      tensor_quantity_extractors_constituents;

    mutable std::vector<dealii::XDMFEntry> xdmfEntries;

    // write output for a tensor quantity of the mixture
    std::pair<
      std::vector<std::string>,
      std::vector<
        dealii::DataComponentInterpretation::DataComponentInterpretation>>
    write_tensor_quantity(
      const std::string             &quantity_name,
      const dealii::DoFHandler<dim> &dof_handler,
      const dealii::CellDataStorage<
        typename dealii::Triangulation<dim>::cell_iterator,
        LocalMixture<dim, Number>>                       &local_mixtures,
      const QuantityExtractor<dim, Number>               &extractor,
      const dealii::DoFHandler<dim>                      &helper_dof_handler,
      dealii::LinearAlgebra::distributed::Vector<Number> &helper_vector) const;

    // write output for a scalar quantity of the mixture
    std::pair<
      std::vector<std::string>,
      std::vector<
        dealii::DataComponentInterpretation::DataComponentInterpretation>>
    write_scalar_quantity(
      const std::string             &quantity_name,
      const dealii::DoFHandler<dim> &dof_handler,
      const dealii::CellDataStorage<
        typename dealii::Triangulation<dim>::cell_iterator,
        LocalMixture<dim, Number>>                       &local_mixtures,
      const QuantityExtractor<dim, Number>               &extractor,
      const dealii::DoFHandler<dim>                      &helper_dof_handler,
      dealii::LinearAlgebra::distributed::Vector<Number> &helper_vector) const;

    // write output for a scalar quantity on constituent level
    std::pair<
      std::vector<std::string>,
      std::vector<
        dealii::DataComponentInterpretation::DataComponentInterpretation>>
    write_scalar_quantity_constituent(
      const std::string             &quantity_name,
      const unsigned int             constituent_id,
      const dealii::DoFHandler<dim> &dof_handler,
      const dealii::CellDataStorage<
        typename dealii::Triangulation<dim>::cell_iterator,
        LocalMixture<dim, Number>>                       &local_mixtures,
      const QuantityExtractorConstituent<dim, Number>    &extractor,
      const dealii::DoFHandler<dim>                      &helper_dof_handler,
      dealii::LinearAlgebra::distributed::Vector<Number> &helper_vector) const;

    // write output for a vector quantity on constituent level
    std::pair<
      std::vector<std::string>,
      std::vector<
        dealii::DataComponentInterpretation::DataComponentInterpretation>>
    write_vector_quantity_constituent(
      const std::string             &quantity_name,
      const unsigned int             constituent_id,
      const dealii::DoFHandler<dim> &dof_handler,
      const dealii::CellDataStorage<
        typename dealii::Triangulation<dim>::cell_iterator,
        LocalMixture<dim, Number>>                       &local_mixtures,
      const QuantityExtractorConstituent<dim, Number>    &extractor,
      const dealii::DoFHandler<dim>                      &helper_dof_handler,
      dealii::LinearAlgebra::distributed::Vector<Number> &helper_vector) const;

    // write output for a tensor quantity on constituent level
    std::pair<
      std::vector<std::string>,
      std::vector<
        dealii::DataComponentInterpretation::DataComponentInterpretation>>
    write_tensor_quantity_constituent(
      const std::string             &quantity_name,
      const unsigned int             constituent_id,
      const dealii::DoFHandler<dim> &dof_handler,
      const dealii::CellDataStorage<
        typename dealii::Triangulation<dim>::cell_iterator,
        LocalMixture<dim, Number>>                       &local_mixtures,
      const QuantityExtractorConstituent<dim, Number>    &extractor,
      const dealii::DoFHandler<dim>                      &helper_dof_handler,
      dealii::LinearAlgebra::distributed::Vector<Number> &helper_vector) const;

    void
    write_output_base(
      const dealii::DoFHandler<dim> &dof_handler,
      const dealii::CellDataStorage<
        typename dealii::Triangulation<dim>::cell_iterator,
        LocalMixture<dim, Number>>                       &local_mixtures,
      const QuantityExtractor<dim, Number>               &extractor,
      const dealii::DoFHandler<dim>                      &helper_dof_handler,
      dealii::LinearAlgebra::distributed::Vector<Number> &helper_vector) const;

    void
    write_output_base_constituent(
      const unsigned int             constituent_id,
      const dealii::DoFHandler<dim> &dof_handler,
      const dealii::CellDataStorage<
        typename dealii::Triangulation<dim>::cell_iterator,
        LocalMixture<dim, Number>>                       &local_mixtures,
      const QuantityExtractorConstituent<dim, Number>    &extractor,
      const dealii::DoFHandler<dim>                      &helper_dof_handler,
      dealii::LinearAlgebra::distributed::Vector<Number> &helper_vector) const;
  };
} // namespace Mixture::Postprocessing
