#pragma once

#include <MSHCMM/mixture/constituents/cylindrical_coordinate_transformer.h>

#include <functional>
#include <memory>
#include <vector>

#include "local_mixture.h"

namespace Mixture::Postprocessing
{

  template <int dim, typename Number = double>
  using QuantityExtractor = std::function<
    void(const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                                 local_mixture,
         dealii::Vector<Number> &vector)>;

  template <int dim, typename Number = double>
  using QuantityExtractorConstituent = std::function<
    void(const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                                 local_mixture,
         const unsigned int      constituent_id,
         dealii::Vector<Number> &vector)>;

  template <int dim, typename Number = double>
  class GLStrain_v2
  {
  public:
    void
    operator()(
      const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                              local_mixture,
      dealii::Vector<Number> &vector) const
    {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // cell_rhs
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get deformation gradient of mixture
          const auto &F = local_mixture[q]->get_mixture_deformation_gradient();
          // compute Green-Lagrange strain
          const auto E = dealii::Physics::Elasticity::Kinematics::E(F);

          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              vector(q + (3 * i + j) * n_quad_points) = E[i][j];
        }
    };
  };


  template <int dim, typename Number = double>
  class RotatedCauchyStress
  {
  public:
    explicit RotatedCauchyStress(
      const Constituents::CylindricalCoordinateTransformer<dim>
        &cos_transformer)
      : cos_transformer(cos_transformer)
    {}

    void
    operator()(
      const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                              local_mixture,
      dealii::Vector<Number> &vector) const
    {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // cell_rhs
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get deformation gradient of mixture
          const auto &F = local_mixture[q]->get_mixture_deformation_gradient();

          // get second Piola-Kirchhoff stress of mixture
          const auto &PK2_stress = local_mixture[q]->get_mixture_stress();

          // transform PK2 stress to Cauchy stress
          const auto sigma =
            1.0 / determinant(F) * F * PK2_stress * transpose(F);

          // get current location
          const auto &p = local_mixture[q]->get_location();
          // rotate stress
          const auto sigma_rotated = cos_transformer.from_cartesian(p, sigma);

          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              vector(q + (3 * i + j) * n_quad_points) = sigma_rotated[i][j];
        }
    }

  private:
    const Constituents::CylindricalCoordinateTransformer<dim> &cos_transformer;
  };


  template <int dim, typename Number = double>
  class RotatedPK1Stress
  {
  public:
    explicit RotatedPK1Stress(
      const Constituents::CylindricalCoordinateTransformer<dim>
        &cos_transformer)
      : cos_transformer(cos_transformer)
    {}

    void
    operator()(
      const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                              local_mixture,
      dealii::Vector<Number> &vector) const
    {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // cell_rhs
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get deformation gradient of mixture
          const auto &F = local_mixture[q]->get_mixture_deformation_gradient();

          // get second Piola-Kirchhoff stress of mixture
          // this does include the isochoric and volumetric contributions of the
          // growth strategy in the case of the IncompressibleNeoHookeDecoupled
          // material
          const auto &PK2_stress = local_mixture[q]->get_mixture_stress();

          // transform PK2 stress to PK1 stress
          const auto PK1 = F * PK2_stress;

          // get current location
          const auto &p = local_mixture[q]->get_location();
          // rotate stress
          const auto PK1_rotated = cos_transformer.from_cartesian(p, PK1);

          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              vector(q + (3 * i + j) * n_quad_points) = PK1_rotated[i][j];
        }
    }

  private:
    const Constituents::CylindricalCoordinateTransformer<dim> &cos_transformer;
  };


  template <int dim, typename Number = double>
  class RotatedDefGrad
  {
  public:
    explicit RotatedDefGrad(
      const Constituents::CylindricalCoordinateTransformer<dim>
        &cos_transformer)
      : cos_transformer(cos_transformer)
    {}

    void
    operator()(
      const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                              local_mixture,
      dealii::Vector<Number> &vector) const
    {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // cell_rhs
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get deformation gradient of mixture
          const auto &F = local_mixture[q]->get_mixture_deformation_gradient();

          // get current location
          const auto &p = local_mixture[q]->get_location();
          // rotate stress
          const auto F_rotated = cos_transformer.from_cartesian(p, F);

          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              vector(q + (3 * i + j) * n_quad_points) = F_rotated[i][j];
        }
    }

  private:
    const Constituents::CylindricalCoordinateTransformer<dim> &cos_transformer;
  };


  template <int dim, typename Number = double>
  class RotatedPK2Stress
  {
  public:
    explicit RotatedPK2Stress(
      const Constituents::CylindricalCoordinateTransformer<dim>
        &cos_transformer)
      : cos_transformer(cos_transformer)
    {}

    void
    operator()(
      const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                              local_mixture,
      dealii::Vector<Number> &vector) const
    {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // cell_rhs
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get second Piola-Kirchhoff stress of mixture
          // this does include the isochoric and volumetric contributions of the
          // growth strategy
          const auto &PK2_stress = local_mixture[q]->get_mixture_stress();

          // get current location
          const auto &p = local_mixture[q]->get_location();
          // rotate stress
          const auto PK2_rotated =
            cos_transformer.from_cartesian(p, PK2_stress);

          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              vector(q + (3 * i + j) * n_quad_points) = PK2_rotated[i][j];
        }
    }

  private:
    const Constituents::CylindricalCoordinateTransformer<dim> &cos_transformer;
  };


  template <int dim, typename Number = double>
  class RotatedPrestretch
  {
  public:
    explicit RotatedPrestretch(
      const Constituents::CylindricalCoordinateTransformer<dim>
        &cos_transformer)
      : cos_transformer(cos_transformer)
    {}

    void
    operator()(
      const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                              local_mixture,
      const unsigned int      constituent_id,
      dealii::Vector<Number> &vector) const
    {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // cell_rhs
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get prestretch tensor of hyperelastic constituent
          const auto &G =
            local_mixture[q]->get_constituent_prestretch_tensor(constituent_id);

          // get current location
          const auto &p = local_mixture[q]->get_location();
          // rotate stress
          const auto G_rotated = cos_transformer.from_cartesian(p, G);

          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              vector(q + (3 * i + j) * n_quad_points) = G_rotated[i][j];
        }
    }

  private:
    const Constituents::CylindricalCoordinateTransformer<dim> &cos_transformer;
  };


  template <int dim, typename Number = double>
  const QuantityExtractor<dim, Number> GLStrain =
    [](const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                               local_mixture,
       dealii::Vector<Number> &vector) {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // cell_rhs
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get deformation gradient of mixture
          const auto &F = local_mixture[q]->get_mixture_deformation_gradient();
          // compute Green-Lagrange strain
          const auto E = dealii::Physics::Elasticity::Kinematics::E(F);

          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              vector(q + (3 * i + j) * n_quad_points) = E[i][j];
        }
    };

  /**
   * Deformation gradient
   * @tparam dim
   * @tparam Number
   */
  template <int dim, typename Number = double>
  const QuantityExtractor<dim, Number> DeformationGradient =
    [](const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                               local_mixture,
       dealii::Vector<Number> &vector) {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // cell_rhs
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get deformation gradient of mixture
          const auto &F = local_mixture[q]->get_mixture_deformation_gradient();

          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              vector(q + (3 * i + j) * n_quad_points) = F[i][j];
        }
    };


  template <int dim, typename Number = double>
  const QuantityExtractor<dim, Number> PK2Stress =
    [](const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                               local_mixture,
       dealii::Vector<Number> &vector) {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // cell_rhs
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get second Piola-Kirchhoff stress of mixture
          const auto &PK2_stress = local_mixture[q]->get_mixture_stress();

          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              vector(q + (3 * i + j) * n_quad_points) = PK2_stress[i][j];
        }
    };

  template <int dim, typename Number = double>
  const QuantityExtractor<dim, Number> volumetric_PK2Stress =
    [](const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                               local_mixture,
       dealii::Vector<Number> &vector) {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // cell_rhs
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get volumetric second Piola-Kirchhoff stress of mixture coming from
          // growth strategy
          const auto &PK2_stress_vol =
            local_mixture[q]->get_mixture_volumetric_stress();

          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              vector(q + (3 * i + j) * n_quad_points) = PK2_stress_vol[i][j];
        }
    };

  template <int dim, typename Number = double>
  const QuantityExtractor<dim, Number> CauchyStress =
    [](const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                               local_mixture,
       dealii::Vector<Number> &vector) {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // cell_rhs
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get deformation gradient of mixture
          const auto &F = local_mixture[q]->get_mixture_deformation_gradient();

          // get second Piola-Kirchhoff stress of mixture
          const auto &PK2_stress = local_mixture[q]->get_mixture_stress();

          // transform PK2 stress to Cauchy stress
          const auto sigma =
            1.0 / determinant(F) * F * PK2_stress * transpose(F);

          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              vector(q + (3 * i + j) * n_quad_points) = sigma[i][j];
        }
    };

  template <int dim, typename Number = double>
  const QuantityExtractor<dim, Number> PK1Stress =
    [](const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                               local_mixture,
       dealii::Vector<Number> &vector) {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // cell_rhs
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get deformation gradient of mixture
          const auto &F = local_mixture[q]->get_mixture_deformation_gradient();

          // get second Piola-Kirchhoff stress of mixture
          const auto &PK2_stress = local_mixture[q]->get_mixture_stress();

          // transform PK2 stress to first Piola Kirchhoff stress
          const auto PK1 = F * PK2_stress;

          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              vector(q + (3 * i + j) * n_quad_points) = PK1[i][j];
        }
    };


  template <int dim, typename Number = double>
  const QuantityExtractor<dim, Number> detF_mixture =
    [](const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                               local_mixture,
       dealii::Vector<Number> &vector) {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // vector
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get deformation gradient of mixture
          const auto &F = local_mixture[q]->get_mixture_deformation_gradient();
          // compute determinant
          vector(q) = dealii::determinant(F);
        }
    };


  template <int dim, typename Number = double>
  const QuantityExtractor<dim, Number> trace_sigma_mixture =
    [](const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                               local_mixture,
       dealii::Vector<Number> &vector) {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // cell_rhs
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get deformation gradient of mixture
          const auto &F = local_mixture[q]->get_mixture_deformation_gradient();

          // get second Piola-Kirchhoff stress of mixture
          const auto &PK2_stress = local_mixture[q]->get_mixture_stress();

          // transform PK2 stress to Cauchy stress
          const auto sigma =
            1.0 / determinant(F) * F * PK2_stress * transpose(F);
          // compute and store trace of sigma
          vector(q) = dealii::trace(sigma);
        }
    };


  template <int dim, typename Number = double>
  const QuantityExtractorConstituent<dim, Number> CurrentMassFractionRatio =
    [](const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                               local_mixture,
       const unsigned int      constituent_id,
       dealii::Vector<Number> &vector) {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // vector
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get current reference growth scalar of constituent
          vector(q) = local_mixture[q]->get_constituent_mass_fraction_ratio(
            constituent_id);
        }
    };

  template <int dim, typename Number = double>
  const QuantityExtractorConstituent<dim, Number> CurrentLambdaR =
    [](const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                               local_mixture,
       const unsigned int      constituent_id,
       dealii::Vector<Number> &vector) {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // vector
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get current reference growth scalar of constituent
          vector(q) = local_mixture[q]
                        ->get_transferable_parameters(constituent_id)
                        .lambda_rem;
        }
    };

  template <int dim, typename Number = double>
  const QuantityExtractorConstituent<dim, Number> PK2StressConstituent =
    [](const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                               local_mixture,
       const unsigned int      constituent_id,
       dealii::Vector<Number> &vector) {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // cell_rhs
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get second Piola-Kirchhoff stress of mixture
          // todo: is this the correct stress? This does not include the
          // volumetric stress contributions...
          const auto &PK2_stress =
            local_mixture[q]->get_constituent_stress_mixture_level(
              constituent_id);

          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              vector(q + (3 * i + j) * n_quad_points) = PK2_stress[i][j];
        }
    };

  template <int dim, typename Number = double>
  const QuantityExtractorConstituent<dim, Number> PrestretchTensorConstituent =
    [](const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                               local_mixture,
       const unsigned int      constituent_id,
       dealii::Vector<Number> &vector) {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // cell_rhs
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get second Piola-Kirchhoff stress of mixture
          const auto &prestretch_tensor =
            local_mixture[q]->get_constituent_prestretch_tensor(constituent_id);

          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              vector(q + (3 * i + j) * n_quad_points) = prestretch_tensor[i][j];
        }
    };

  //! @brief Get the fiber Cauchy stress. Note this only makes sense for fiber constituents!
  template <int dim, typename Number = double>
  const QuantityExtractorConstituent<dim, Number> FiberCauchyStressConstituent =
    [](const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                               local_mixture,
       const unsigned int      constituent_id,
       dealii::Vector<Number> &vector) {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // cell_rhs
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get Cauchy stress of fiber
          const auto &transferable_parameters =
            local_mixture[q]->get_transferable_parameters(constituent_id);
          vector(q) = transferable_parameters.fiber_cauchy_stress;
        }
    };

  //! @brief Get the fiber Cauchy stress. Note this only makes sense for fiber constituents!
  template <int dim, typename Number = double>
  const QuantityExtractorConstituent<dim, Number>
    ActiveFiberCauchyStressConstituent =
      [](const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                                 local_mixture,
         const unsigned int      constituent_id,
         dealii::Vector<Number> &vector) {
        // get size of mixture, i.e. number of quadrature points
        const unsigned int n_quad_points = local_mixture.size();
        // loop over quadrature points and assemble extracted quantities into
        // cell_rhs
        for (unsigned int q = 0; q < n_quad_points; ++q)
          {
            // get Cauchy stress of fiber
            const auto &transferable_parameters =
              local_mixture[q]->get_transferable_parameters(constituent_id);
            vector(q) = transferable_parameters.active_fiber_cauchy_stress;
          }
      };

  //! @brief Get the homeostatic fiber Cauchy stress. Note this only makes sense for fiber constituents!
  template <int dim, typename Number = double>
  const QuantityExtractorConstituent<dim, Number>
    HomeostaticCauchyStressConstituent =
      [](const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                                 local_mixture,
         const unsigned int      constituent_id,
         dealii::Vector<Number> &vector) {
        // get size of mixture, i.e. number of quadrature points
        const unsigned int n_quad_points = local_mixture.size();
        // loop over quadrature points and assemble extracted quantities into
        // cell_rhs
        for (unsigned int q = 0; q < n_quad_points; ++q)
          {
            // get Cauchy stress of fiber
            const auto &transferable_parameters =
              local_mixture[q]->get_transferable_parameters(constituent_id);
            vector(q) = transferable_parameters.sigma_hom;
          }
      };

  //! @brief Get the orientation of the fibers. Note this only makes sense for fiber constituents!
  template <int dim, typename Number = double>
  const QuantityExtractorConstituent<dim, Number> FiberOrientation =
    [](const std::vector<std::shared_ptr<const LocalMixture<dim, Number>>>
                               local_mixture,
       const unsigned int      constituent_id,
       dealii::Vector<Number> &vector) {
      // get size of mixture, i.e. number of quadrature points
      const unsigned int n_quad_points = local_mixture.size();
      // loop over quadrature points and assemble extracted quantities into
      // cell_rhs
      for (unsigned int q = 0; q < n_quad_points; ++q)
        {
          // get second Piola-Kirchhoff stress of mixture
          const auto &fiber_direction =
            local_mixture[q]->get_constituent_fiber_direction(constituent_id);

          for (unsigned int i = 0; i < dim; ++i)
            vector(q + i * n_quad_points) = fiber_direction[i];
        }
    };

} // namespace Mixture::Postprocessing
