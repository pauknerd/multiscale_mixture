#pragma once

#include <MSHCMM/mixture/prestretch_strategies/prestretch_strategy_base.h>

#include "MSHCMM/mixture/constituents/constituent_fiber.h"
#include "MSHCMM/mixture/constituents/constituent_hyperelastic.h"
#include "unordered_map"

namespace Mixture
{
  /**
   * @class LocalMixture
   *
   * @brief Stores all the constituents present at a certain quadrature point.
   *
   * This class wraps all the constituents at a single quadrature point. An
   * instance of this class is stored at every quadrature point and handles
   * everything related to the solid mechanics.
   *
   * @tparam dim
   * @tparam Number
   */

  template <int dim, typename Number = double>
  class LocalMixture
  {
  public:
    /**
     * @brief Setup all the necessary quantities for the local mixture on a quadrature point. Note that
     * `initial_mass_frac` and `constituents_in` must have the same length.
     * Also, note that it is assumed that the entries in `initial_mass_frac` and
     * `constituents_in` correspond to each other, i.e., the first entry in
     * `initial_mass_frac` represents the initial mass fraction of the first
     * constituent in `constituents_in`.
     *
     * @param[in] quad_point_location Spatial coordinates of quadrature point
     * @param[in] initial_ref_density Initial reference density of the entire
     * mixture
     * @param[in] initial_mass_frac Vector containing the initial mass fractions
     * of all the constituents of the local mixture.
     * @param[in] constituents_in Vector with all the constituents present at
     * the local mixture
     */
    void
    setup_local_data(
      const dealii::Point<dim>  &quad_point_location,
      const Number               initial_ref_density,
      const std::vector<Number> &initial_mass_frac,
      std::vector<std::unique_ptr<Constituents::ConstituentBase<dim, Number>>>
        constituents_in);

    void
    print_constituents();

    /**
     * Update the data of all constituents in this local mixture based on the
     * mixture-level deformation gradient `F` and the inverse of the growth
     * deformation gradient `F_g_inv`. The parameters `S_vol` and `C_vol`
     * represent the volumetric stress (second Piola-Kirchoff) and volumetric
     * tangent contribution, respectively. The parameters `p` and `J` are the
     * pressure and the determinant of F, respectively. Note that they are part
     * of the three-field formulation.
     *
     * @param[in] F Mixture-level deformation gradient
     * @param[in] F_g_inv Inverse of mixture-level growth deformation gradient
     * @param[in] S_vol 2nd Piola-Kirchhoff stress contribution of volumetric
     * part (computed by GrowthStrategy)
     * @param[in] C_vol Tangent contribution of volumetric part (computed by
     * GrowthStrategy)
     *
     */
    void
    update_values(const dealii::Tensor<2, dim, Number>          &F,
                  const dealii::Tensor<2, dim, Number>          &F_g_inv,
                  const dealii::SymmetricTensor<2, dim, Number> &S_vol,
                  const dealii::SymmetricTensor<4, dim, Number> &C_vol);

    /**
     * @brief Update constituent data for a selected number of constituents.
     *
     * @param constituent_ids Constituents to update.
     * @param[in] F Mixture-level deformation gradient
     * @param[in] F_g_inv Inverse of mixture-level growth deformation gradient
     * @param[in] S_vol 2nd Piola-Kirchhoff stress contribution of volumetric
     * part (computed by GrowthStrategy)
     * @param[in] C_vol Tangent contribution of volumetric part (computed by
     * GrowthStrategy)
     */
    void
    update_values(const std::vector<unsigned int>      &constituent_ids,
                  const dealii::Tensor<2, dim, Number> &F,
                  const dealii::Tensor<2, dim, Number> &F_g_inv,
                  const dealii::SymmetricTensor<2, dim, Number> &S_vol,
                  const dealii::SymmetricTensor<4, dim, Number> &C_vol);

    /**
     * Update growth and remodeling of all the constituents stored in the local
     * mixture. This also updated the `current_mass_fraction_ratio` at the
     * quadrature point. This function is called at the beginning of each time
     * step.
     *
     * @param F_g_inv Inverse of growth deformation gradient
     * @param coupled Flag indicating of the problem is coupled to diffusion and pathways. If false,
     * the normal homogenized constrained mixture model is used.
     */
    void
    update_GR(const dealii::Tensor<2, dim, Number> &F_g_inv,
              const bool                            coupled);

    /**
     * @brief Update the prestretch of constituents with specified ids.
     *
     * @details
     * This function updates the prestretch of the constituents with ids in
     * `constituent_ids`. Note that fiber-based constituents don't do anything
     * in that case, this function is only needed when updating a hyperelastic
     * constituent iteratively during the prestressing algorithm. Therefore, it
     * makes sense to only pass the constituent_id of hyperelastic constituent
     * that is iteratively prestressed as an argument.
     *
     * @param constituent_ids Vector with all the constituent_ids to update.
     * @param F Current deformation gradient at mixture level.
     */
    void
    update_prestretch(const std::vector<unsigned int>      &constituent_ids,
                      const dealii::Tensor<2, dim, Number> &F,
                      const dealii::Point<dim, Number>     &p,
                      PrestretchStrategies::PrestretchStrategyBase<dim, Number>
                        &prestretch_strategy);

    /**
     * @brief Evaluate and set the prestretch of all constituents based on the prestretch function
     * that they store. Note that the time in the prestretch functions is set by
     * the
     * @class Mixture_GR_ThreeField class.
     *
     * @param point Spatial coordinates of the local mixture.
     */
    void
    evaluate_prestretch(const dealii::Point<dim> &point);

    /**
     * @brief Get the number of constituents in the local mixture.
     * @returns Integer.
     */
    [[nodiscard]] unsigned int
    n_constituents() const;

    /**
     * @brief Check if a constituent with the given id is present in the local mixture.
     */
    bool
    constituent_present(const unsigned int constituent_id) const;

    /**
     * @brief Get all the constituents of the local mixture.
     *
     * @return A vector with all the constituents at the local mixture.
     */
    const std::unordered_map<
      unsigned int,
      std::unique_ptr<Constituents::ConstituentBase<dim, Number>>> &
    get_constituents() const;

    /**
     * @brief Same as above but non-const.
     *
     * @return A vector with all the constituents at the local mixture.
     */
    std::unordered_map<
      unsigned int,
      std::unique_ptr<Constituents::ConstituentBase<dim, Number>>> &
    get_constituents();

    /**
     * @brief Get a constituent with the specified `constituent_id`.
     *
     * @param[in] constituent_id Id of the constituent to return.
     *
     * @returns A unique pointer to the constituent with id `constituent_id``
     */
    const std::unique_ptr<Constituents::ConstituentBase<dim, Number>> &
    get_constituent(const unsigned int constituent_id) const;

    /**
     * @brief Same as above but non-const.
     *
     * @param[in] constituent_id Id of the constituent to return.
     *
     * @returns A unique pointer to the constituent with id `constituent_id``
     */
    std::unique_ptr<Constituents::ConstituentBase<dim, Number>> &
    get_constituent(const unsigned int constituent_id);

    /**
     * @brief Get the `current_mass_fraction_ratio` of the local mixture.
     *
     * @returns A scalar representing the factor by how much the volume has increased.
     */
    Number
    get_current_mass_fraction_ratio() const;

    Number
    get_initial_reference_density() const;

    const std::unordered_map<unsigned int, Number> &
    get_initial_mass_fractions() const;

    Number
    get_initial_mass_fraction(const unsigned int constituent_id) const;

    const dealii::Tensor<2, dim, Number> &
    get_mixture_deformation_gradient() const;

    // get PK2 stress at the mixture level
    const dealii::SymmetricTensor<2, dim, Number> &
    get_mixture_stress() const;

    const dealii::SymmetricTensor<2, dim, Number> &
    get_mixture_volumetric_stress() const;

    const dealii::SymmetricTensor<4, dim, Number> &
    get_mixture_tangent() const;

    Number
    get_constituent_mass_fraction_ratio(
      const unsigned int constituent_id) const;

    // todo: not sure if it is ok to return a reference here...if I do, compiler
    // prints a warning, it still works though
    //! @brief constituent stress at mixture level
    dealii::SymmetricTensor<2, dim, Number>
    get_constituent_stress_mixture_level(
      const unsigned int constituent_id) const;

    // todo: those function might all be removed and replaced with
    // get_constituent(constituent_id)->get_XXX()
    //! @brief constituent stress at constituent level
    const dealii::SymmetricTensor<2, dim, Number> &
    get_constituent_stress(const unsigned int constituent_id) const;

    const dealii::SymmetricTensor<4, dim, Number> &
    get_constituent_tangent(const unsigned int constituent_id) const;

    const Constituents::TransferableParameters<Number> &
    get_transferable_parameters(const unsigned int constituent_id) const;

    Constituents::TransferableParameters<Number> &
    get_transferable_parameters(const unsigned int constituent_id);

    const dealii::Tensor<1, dim, Number> &
    get_constituent_fiber_direction(const unsigned int constituent_id) const;

    const dealii::Tensor<2, dim, Number> &
    get_constituent_prestretch_tensor(const unsigned int constituent_id) const;

    /**
     * @brief Get the spatial coordinates of the local mixture.
     *
     * @details
     * This function is mainly used in the Postprocessing functions inc ase some
     * kind of spatial dependent quantities are needed, such as a coordinate
     * transformation.
     *
     * @returns Point<dim>.
     */
    const dealii::Point<dim> &
    get_location() const;

  private:
    // coordinates of local mixture, needed for postprocessing in cylindrical
    // coordinate system
    dealii::Point<dim> location;
    // initial reference density of the mixture, remains constant
    Number initial_reference_density;
    // initial mass fractions of the constituents
    std::unordered_map<unsigned int, Number> initial_mass_fractions;
    // Constituents at the quadrature point
    std::unordered_map<
      unsigned int,
      std::unique_ptr<Constituents::ConstituentBase<dim, Number>>>
      constituents;

    // current reference mass fraction ratio, depends on the individual
    // mass_fraction_ratios of the constituents and on total mass density of
    // mixture
    Number current_mass_fraction_ratio = 1.0;

    // mixture-level deformation gradient
    dealii::Tensor<2, dim, Number> F_mixture{
      dealii::Physics::Elasticity::StandardTensors<dim>::I};
    // mixture-level stress, PK2
    dealii::SymmetricTensor<2, dim, Number> mixture_stress;
    // mixture-level volumetric stress, PK2 stress from growth strategy
    dealii::SymmetricTensor<2, dim, Number> mixture_volumetric_stress;
    // mixture-level stiffness matrix
    dealii::SymmetricTensor<4, dim, Number> mixture_tangent;
  };
} // namespace Mixture
