#include <MSHCMM/mixture/constituents/constituent_hyperelastic.h>
#include <MSHCMM/mixture/constituents/constituent_hyperelastic_factory.h>

namespace Mixture::Constituents
{

  template <int dim, typename Number>
  ConstituentHyperelasticFactory<dim, Number>::ConstituentHyperelasticFactory(
    const unsigned int                      constituent_id,
    dealii::TensorFunction<2, dim, Number> &prestretch_function,
    MaterialCreator                         material_creator,
    const std::vector<unsigned int>        &material_ids)
    : ConstituentFactoryBase<dim, Number>(material_ids)
    , constituent_id(constituent_id)
    , prestretch_function(prestretch_function)
    , material_creator(std::move(material_creator))
  {}

  template <int dim, typename Number>
  std::unique_ptr<ConstituentBase<dim, Number>>
  ConstituentHyperelasticFactory<dim, Number>::create_constituent(
    const dealii::Point<dim, Number> &point)
  {
    return std::make_unique<ConstituentHyperelastic<dim, Number>>(
      constituent_id,
      prestretch_function.value(point),
      prestretch_function,
      material_creator(point));
  }

  template <int dim, typename Number>
  void
  ConstituentHyperelasticFactory<dim, Number>::set_time(const Number time) const
  {
    prestretch_function.set_time(time);
  }

  // instantiations
  template class ConstituentHyperelasticFactory<3, double>;
} // namespace Mixture::Constituents