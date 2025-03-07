#include <MSHCMM/mixture/constituents/constituent_fiber.h>
#include <MSHCMM/mixture/constituents/constituent_fiber_factory.h>

namespace Mixture::Constituents
{

  template <int dim, typename Number>
  ConstituentFiberFactory<dim, Number>::ConstituentFiberFactory(
    const unsigned int               constituent_id,
    dealii::Function<dim, Number>   &prestretch_function,
    FiberMaterialCreator             material_creator,
    const Number                     time_step_size,
    const Number                     decay_time,
    const Number                     gain,
    const std::vector<unsigned int> &material_ids)
    : ConstituentFactoryBase<dim, Number>(material_ids)
    , constituent_id(constituent_id)
    , prestretch_function(prestretch_function)
    , material_creator(std::move(material_creator))
    , dt(time_step_size)
    , decay_time(decay_time)
    , gain(gain)
  {}

  template <int dim, typename Number>
  std::unique_ptr<ConstituentBase<dim, Number>>
  ConstituentFiberFactory<dim, Number>::create_constituent(
    const dealii::Point<dim, Number> &point)
  {
    return std::make_unique<ConstituentFiber<dim, Number>>(
      constituent_id,
      prestretch_function.value(point),
      prestretch_function,
      material_creator(point),
      dt,
      decay_time,
      gain);
  }

  template <int dim, typename Number>
  void
  ConstituentFiberFactory<dim, Number>::set_time(const Number time) const
  {
    prestretch_function.set_time(time);
  }

  // instantiations
  template class ConstituentFiberFactory<3, double>;

} // namespace Mixture::Constituents
