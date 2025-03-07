#pragma once

#include "constituent_base.h"

namespace Mixture::Constituents
{

  template <int dim, typename Number = double>
  class ConstituentFactoryBase
  {
  public:
    //! definition of a std::function to create a material
    using MaterialCreator =
      std::function<std::unique_ptr<Materials::MaterialBase<dim, Number>>(
        const dealii::Point<dim, Number> &point)>;

    //! definition of a std::function to create a fiber material
    using FiberMaterialCreator =
      std::function<std::unique_ptr<Materials::FiberMaterialBase<dim, Number>>(
        const dealii::Point<dim, Number> &point)>;

    virtual ~ConstituentFactoryBase() = default;

    virtual std::unique_ptr<ConstituentBase<dim, Number>>
    create_constituent(const dealii::Point<dim, Number> &point) = 0;

    //! check if constituent should be added to element with element_material_id
    [[nodiscard]] bool
    check_material_id(const unsigned int element_material_id) const
    {
      auto it = std::find(material_ids.cbegin(),
                          material_ids.cend(),
                          element_material_id);
      return it != material_ids.cend();
    }

    //! set time of prestretch functions
    virtual void
    set_time(const Number time) const = 0;

  protected:
    // make constructor protected to avoid instantiations of that class.
    explicit ConstituentFactoryBase(
      const std::vector<unsigned int> &material_ids)
      : material_ids(material_ids)
    {}

  private:
    const std::vector<unsigned int> material_ids;
  };
} // namespace Mixture::Constituents
