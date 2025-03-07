#pragma once

#include "constituent_base.h"
#include "constituent_factory_base.h"

namespace Mixture::Constituents
{
  /**
   * Factory method for fiber constituent.
   *
   * @tparam dim
   * @tparam Number
   */
  template <int dim, typename Number = double>
  class ConstituentFiberFactory : public ConstituentFactoryBase<dim, Number>
  {
  public:
    using typename ConstituentFactoryBase<dim, Number>::FiberMaterialCreator;

    /**
     * @brief Creates a fiber based hyperelastic constituent.
     *
     * @param[in] constituent_id
     * @param[in] prestretch_function
     * @param[in] material_creator Returns the underlying material of the
     * constituent.
     * @param[in] dt Time step size of the explicit time integration scheme
     * (forward Euler) used to update the internal quantities.
     * @param[in] material_ids Can be used to localize the constituent to
     * specific spatial location. Note that the name material_id is based on
     * dealii naming and has nothing to do with the material used for the
     * constituent.
     *
     */
    explicit ConstituentFiberFactory(
      const unsigned int               constituent_id,
      dealii::Function<dim, Number>   &prestretch_function,
      FiberMaterialCreator             material_creator,
      const Number                     time_step_size,
      const Number                     decay_time,
      const Number                     gain,
      const std::vector<unsigned int> &material_ids = {0});

    std::unique_ptr<ConstituentBase<dim, Number>>
    create_constituent(const dealii::Point<dim, Number> &point) override;

    void
    set_time(const Number time) const override;

  private:
    // constituents created by this factory will have this id, important for
    // writing output and to connect a transformer
    unsigned int constituent_id;
    // need reference because of polymorphism, otherwise the value() function
    // of the Function base class is called which is abstract. Note that this
    // allows for spatially varying prestretches
    dealii::Function<dim, Number> &prestretch_function;
    // function that can be used to assign different material parameters at each
    // quadrature point, e.g., fiber orientation.
    FiberMaterialCreator material_creator;
    // time step size used in the simulation
    Number dt;
    // decay time
    Number decay_time;
    // gain parameter
    Number gain;
  };
} // namespace Mixture::Constituents
