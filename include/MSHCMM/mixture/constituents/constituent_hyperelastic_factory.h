#pragma once

#include "constituent_base.h"
#include "constituent_factory_base.h"

namespace Mixture::Constituents
{

  // Factory method for hyperelastic constituent
  template <int dim, typename Number = double>
  class ConstituentHyperelasticFactory
    : public ConstituentFactoryBase<dim, Number>
  {
  public:
    using typename ConstituentFactoryBase<dim, Number>::MaterialCreator;

    /** Creates a hyperelastic constituent with the specified constituent id,
     * prestretch function, and prestretch function. Note that if you use a
     * different coordinate system than a cartesian one, e.g., a cylindrical
     * one, create a custom prestretch function that derives from
     * dealii::TensorFunction and captures an appropriate coordinate
     * transformer.
     *
     * @param[in] constituent_id Unique id of a constituent. It is your
     * responsibility to ensure uniqueness.
     * @param[in] prestretch_function
     * @param[in] material_creator Returns the underlying material of the
     * constituent.
     * @param[in] material_ids Can be used to localize the constituent to
     * specific spatial locations. Note that the name material_id is based on
     * dealii naming and has nothing to do with the material used for the
     * constituent.
     *
     */
    explicit ConstituentHyperelasticFactory(
      const unsigned int                      constituent_id,
      dealii::TensorFunction<2, dim, Number> &prestretch_function,
      MaterialCreator                         material_creator,
      const std::vector<unsigned int>        &material_ids = {0});

    std::unique_ptr<ConstituentBase<dim, Number>>
    create_constituent(const dealii::Point<dim, Number> &point) override;

    void
    set_time(const Number time) const override;

  private:
    // constituents created by this factory will have this id, important for
    // writing output
    unsigned int constituent_id;
    // need reference because of polymorphism, otherwise the value() function
    // of the TensorFunction base class is called which is abstract
    dealii::TensorFunction<2, dim, Number> &prestretch_function;
    MaterialCreator                         material_creator;
  };
} // namespace Mixture::Constituents
