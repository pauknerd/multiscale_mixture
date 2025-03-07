#include <MSHCMM/mixture/growth_and_remodeling/local_mixture.h>
#include <MSHCMM/mixture/prestretch_strategies/prestretch_strategy_base.h>

#include <numeric>

namespace Mixture
{

  template <int dim, typename Number>
  void
  LocalMixture<dim, Number>::setup_local_data(
    const dealii::Point<dim>  &quad_point_location,
    const Number               initial_ref_density,
    const std::vector<Number> &initial_mass_frac,
    std::vector<std::unique_ptr<Constituents::ConstituentBase<dim, Number>>>
      constituents_in)
  {
    Assert(
      (initial_mass_frac.size() == constituents_in.size()),
      dealii::ExcMessage(
        "Number of initial mass fractions and number of constituents does not match!"));
    // set location
    location = quad_point_location;
    // set reference density and mass fractions
    initial_reference_density = initial_ref_density;
    // transfer constituents and initial mass fractions
    auto initial_mass_fraction = initial_mass_frac.begin();
    for (auto &constituent : constituents_in)
      {
        // add initial mass fraction
        initial_mass_fractions.emplace(constituent->get_constituent_id(),
                                       *initial_mass_fraction);
        initial_mass_fraction++;
        // add constituent
        constituents.emplace(constituent->get_constituent_id(),
                             std::move(constituent));
      }
  }

  template <int dim, typename Number>
  void
  LocalMixture<dim, Number>::print_constituents()
  {
    for (const auto &[constituent_id, constituent] : constituents)
      {
        std::cout << constituent->get_name() << std::endl;
      }
  }

  // update all the constituents at this quadrature point
  // todo: this should call the other update_values() function with all
  // constituents
  template <int dim, typename Number>
  inline void
  LocalMixture<dim, Number>::update_values(
    const dealii::Tensor<2, dim, Number>          &F,
    const dealii::Tensor<2, dim, Number>          &F_g_inv,
    const dealii::SymmetricTensor<2, dim, Number> &S_vol,
    const dealii::SymmetricTensor<4, dim, Number> &C_vol)
  {
    // create a vector of all the ids
    std::vector<unsigned int> ids;
    // loop over constituents and extract keys
    ids.reserve(constituents.size());
    for (const auto &[id, constituent] : constituents)
      ids.push_back(id);
    // call other update values function
    update_values(ids, F, F_g_inv, S_vol, C_vol);
  }

  // update all the constituents at this quadrature point
  template <int dim, typename Number>
  inline void
  LocalMixture<dim, Number>::update_values(
    const std::vector<unsigned int>               &constituent_ids,
    const dealii::Tensor<2, dim, Number>          &F,
    const dealii::Tensor<2, dim, Number>          &F_g_inv,
    const dealii::SymmetricTensor<2, dim, Number> &S_vol,
    const dealii::SymmetricTensor<4, dim, Number> &C_vol)
  {
    // do something to deformation gradient? such as inelastic part
    // --> F_inelastic will be given by GrowthStrategy which is stored
    // in the MixtureRule_G&R class. So this class has to compute F_inelastic
    // and then pass it to the update_values() method which takes care of
    // updating the constituents stored in this LocalMixture at the quadrature
    // point

    // store mixture level deformation gradient
    F_mixture = F;

    // store volumetric stress contribution, for output purposes and debug
    mixture_volumetric_stress = S_vol;
    // reset mixture stress and tangent by adding stress and tangent
    // contributions from GrowthStrategy
    mixture_stress =
      /*initial_reference_density * current_mass_fraction_ratio */ S_vol;
    mixture_tangent =
      /*initial_reference_density * current_mass_fraction_ratio */ C_vol;

    // loop over constituents and update them
    for (const auto &i : constituent_ids)
      {
        // update material data of individual constituents
        constituents[i]->update_material(F, F_g_inv);

        // get updated stress and tangents and add to mixture_stress and
        // mixture_tangent
        mixture_stress += initial_reference_density *
                          initial_mass_fractions[i] *
                          constituents[i]->get_mass_fraction_ratio() *
                          constituents[i]->get_stress();

        mixture_tangent += initial_reference_density *
                           initial_mass_fractions[i] *
                           constituents[i]->get_mass_fraction_ratio() *
                           constituents[i]->get_tangent();
      }
  }

  template <int dim, typename Number>
  inline void
  LocalMixture<dim, Number>::update_GR(
    const dealii::Tensor<2, dim, Number> &F_g_inv,
    const bool                            coupled)
  {
    // reset current_mass_fraction_ratio
    current_mass_fraction_ratio = 0.0;

    // loop over constituents and update growth and remodeling
    for (auto &[constituent_id, constituent] : constituents)
      {
        // update growth and remodeling
        constituent->update_GR(F_mixture,
                               F_g_inv,
                               coupled ?
                                 initial_mass_fractions[constituent_id] :
                                 -1.0);
        // update current reference growth scalar because the constituent got
        // updated (new stress) and this should trigger growth and remodeling
        // and therefore, an updated mass fraction ratio
        current_mass_fraction_ratio += initial_mass_fractions[constituent_id] *
                                       constituent->get_mass_fraction_ratio();
      }
  }

  template <int dim, typename Number>
  inline void
  LocalMixture<dim, Number>::update_prestretch(
    const std::vector<unsigned int>      &constituent_ids,
    const dealii::Tensor<2, dim, Number> &F,
    const dealii::Point<dim, Number>     &p,
    PrestretchStrategies::PrestretchStrategyBase<dim, Number>
      &prestretch_strategy)
  {
    // loop over constituents and update prestretch
    for (const auto i : constituent_ids)
      prestretch_strategy.update_prestretch(
        constituents[i]->get_prestretch_tensor(), F, p);
  }

  template <int dim, typename Number>
  inline void
  LocalMixture<dim, Number>::evaluate_prestretch(
    const dealii::Point<dim> &point)
  {
    // loop over constituents, evaluate and set new prestretch
    for (auto &[constituent_id, constituent] : constituents)
      constituent->evaluate_prestretch(point);
  }

  template <int dim, typename Number>
  [[nodiscard]] inline unsigned int
  LocalMixture<dim, Number>::n_constituents() const
  {
    return constituents.size();
  }

  template <int dim, typename Number>
  inline bool
  LocalMixture<dim, Number>::constituent_present(
    const unsigned int constituent_id) const
  {
    return constituents.find(constituent_id) != constituents.end();
  }

  template <int dim, typename Number>
  inline const std::unordered_map<
    unsigned int,
    std::unique_ptr<Constituents::ConstituentBase<dim, Number>>> &
  LocalMixture<dim, Number>::get_constituents() const
  {
    return constituents;
  }

  template <int dim, typename Number>
  inline const std::unique_ptr<Constituents::ConstituentBase<dim, Number>> &
  LocalMixture<dim, Number>::get_constituent(
    const unsigned int constituent_id) const
  {
    Assert(this->constituent_present(constituent_id),
           dealii::ExcMessage("Constituent with id " +
                              std::to_string(constituent_id) +
                              " is not present in the local mixture!"));

    return constituents.at(constituent_id);
  }

  template <int dim, typename Number>
  inline std::unordered_map<
    unsigned int,
    std::unique_ptr<Constituents::ConstituentBase<dim, Number>>> &
  LocalMixture<dim, Number>::get_constituents()
  {
    return constituents;
  }

  template <int dim, typename Number>
  inline std::unique_ptr<Constituents::ConstituentBase<dim, Number>> &
  LocalMixture<dim, Number>::get_constituent(const unsigned int constituent_id)
  {
    Assert(this->constituent_present(constituent_id),
           dealii::ExcMessage("Constituent with id " +
                              std::to_string(constituent_id) +
                              " is not present in the local mixture!"));

    return constituents.at(constituent_id);
  }

  template <int dim, typename Number>
  inline Number
  LocalMixture<dim, Number>::get_current_mass_fraction_ratio() const
  {
    // updated during call to update_GR()
    return current_mass_fraction_ratio;
  }

  template <int dim, typename Number>
  inline Number
  LocalMixture<dim, Number>::get_initial_reference_density() const
  {
    return initial_reference_density;
  }

  template <int dim, typename Number>
  inline const std::unordered_map<unsigned int, Number> &
  LocalMixture<dim, Number>::get_initial_mass_fractions() const
  {
    return initial_mass_fractions;
  }

  template <int dim, typename Number>
  inline Number
  LocalMixture<dim, Number>::get_initial_mass_fraction(
    const unsigned int constituent_id) const
  {
    Assert(this->constituent_present(constituent_id),
           dealii::ExcMessage("Constituent with id " +
                              std::to_string(constituent_id) +
                              " is not present in the local mixture!"));

    return initial_mass_fractions.at(constituent_id);
  }

  template <int dim, typename Number>
  inline const dealii::Tensor<2, dim, Number> &
  LocalMixture<dim, Number>::get_mixture_deformation_gradient() const
  {
    return F_mixture;
  }

  template <int dim, typename Number>
  inline const dealii::SymmetricTensor<2, dim, Number> &
  LocalMixture<dim, Number>::get_mixture_stress() const
  {
    return mixture_stress;
  }

  template <int dim, typename Number>
  inline const dealii::SymmetricTensor<2, dim, Number> &
  LocalMixture<dim, Number>::get_mixture_volumetric_stress() const
  {
    return mixture_volumetric_stress;
  }

  template <int dim, typename Number>
  inline const dealii::SymmetricTensor<4, dim, Number> &
  LocalMixture<dim, Number>::get_mixture_tangent() const
  {
    return mixture_tangent;
  }

  template <int dim, typename Number>
  inline Number
  LocalMixture<dim, Number>::get_constituent_mass_fraction_ratio(
    const unsigned int constituent_id) const
  {
    Assert(this->constituent_present(constituent_id),
           dealii::ExcMessage("Constituent with id " +
                              std::to_string(constituent_id) +
                              " is not present in the local mixture!"));

    return constituents.at(constituent_id)->get_mass_fraction_ratio();
  }

  template <int dim, typename Number>
  inline dealii::SymmetricTensor<2, dim, Number>
  LocalMixture<dim, Number>::get_constituent_stress_mixture_level(
    const unsigned int constituent_id) const
  {
    Assert(this->constituent_present(constituent_id),
           dealii::ExcMessage("Constituent with id " +
                              std::to_string(constituent_id) +
                              " is not present in the local mixture!"));

    return initial_reference_density *
           initial_mass_fractions.at(constituent_id) *
           constituents.at(constituent_id)->get_mass_fraction_ratio() *
           constituents.at(constituent_id)->get_stress();
  }

  template <int dim, typename Number>
  inline const dealii::SymmetricTensor<2, dim, Number> &
  LocalMixture<dim, Number>::get_constituent_stress(
    const unsigned int constituent_id) const
  {
    Assert(this->constituent_present(constituent_id),
           dealii::ExcMessage("Constituent with id " +
                              std::to_string(constituent_id) +
                              " is not present in the local mixture!"));

    return constituents.at(constituent_id)->get_stress();
  }

  template <int dim, typename Number>
  inline const dealii::SymmetricTensor<4, dim, Number> &
  LocalMixture<dim, Number>::get_constituent_tangent(
    const unsigned int constituent_id) const
  {
    Assert(this->constituent_present(constituent_id),
           dealii::ExcMessage("Constituent with id " +
                              std::to_string(constituent_id) +
                              " is not present in the local mixture!"));

    return constituents.at(constituent_id)->get_tangent();
  }

  template <int dim, typename Number>
  inline const Constituents::TransferableParameters<Number> &
  LocalMixture<dim, Number>::get_transferable_parameters(
    const unsigned int constituent_id) const
  {
    Assert(this->constituent_present(constituent_id),
           dealii::ExcMessage("Constituent with id " +
                              std::to_string(constituent_id) +
                              " is not present in the local mixture!"));
    return constituents.at(constituent_id)->get_transferable_parameters();
  }

  template <int dim, typename Number>
  inline Constituents::TransferableParameters<Number> &
  LocalMixture<dim, Number>::get_transferable_parameters(
    const unsigned int constituent_id)
  {
    Assert(this->constituent_present(constituent_id),
           dealii::ExcMessage("Constituent with id " +
                              std::to_string(constituent_id) +
                              " is not present in the local mixture!"));
    return constituents.at(constituent_id)->get_transferable_parameters();
  }

  template <int dim, typename Number>
  inline const dealii::Tensor<1, dim, Number> &
  LocalMixture<dim, Number>::get_constituent_fiber_direction(
    const unsigned int constituent_id) const
  {
    Assert(this->constituent_present(constituent_id),
           dealii::ExcMessage("Constituent with id " +
                              std::to_string(constituent_id) +
                              " is not present in the local mixture!"));

    return constituents.at(constituent_id)->get_fiber_direction();
  }

  template <int dim, typename Number>
  inline const dealii::Tensor<2, dim, Number> &
  LocalMixture<dim, Number>::get_constituent_prestretch_tensor(
    const unsigned int constituent_id) const
  {
    Assert(this->constituent_present(constituent_id),
           dealii::ExcMessage("Constituent with id " +
                              std::to_string(constituent_id) +
                              " is not present in the local mixture!"));

    return constituents.at(constituent_id)->get_prestretch_tensor();
  }

  template <int dim, typename Number>
  inline const dealii::Point<dim> &
  LocalMixture<dim, Number>::get_location() const
  {
    return location;
  }


  // instantiations
  template class LocalMixture<2, double>;
  template class LocalMixture<3, double>;

} // namespace Mixture