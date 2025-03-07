#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <MSHCMM/common/BoundaryConditionFunctions.h>
#include <MSHCMM/mixture/constituents/constituent_fiber_factory.h>
#include <MSHCMM/mixture/constituents/constituent_hyperelastic.h>
#include <MSHCMM/mixture/constituents/constituent_hyperelastic_factory.h>
#include <MSHCMM/mixture/growth_and_remodeling/mixture_G&R.h>
#include <MSHCMM/mixture/growth_strategies/growth_strategy_anisotropic.h>
#include <MSHCMM/mixture/materials/fiber_material_1D.h>
#include <MSHCMM/mixture/materials/neo_hooke_compressible.h>
#include <MSHCMM/mixture/prestretch_strategies/prestretch_strategy_mousavi.h>
#include <MSHCMM/mixture/prestretch_strategies/prestretch_strategy_none.h>

#include <iostream>
#include <memory>

template <int dim, typename Number = double>
class ElastinPrestretch : public dealii::TensorFunction<2, dim, Number>
{
public:
  ElastinPrestretch(const dealii::Tensor<2, dim, Number> G_0,
                    const unsigned int                   n_load_steps)
    : dealii::TensorFunction<2, dim, Number>()
    , G(G_0)
    , n_load_steps(n_load_steps)
  {}

  dealii::Tensor<2, dim, Number>
  value(const dealii::Point<dim> &p) const override
  {
    (void)p;

    if (this->get_time() > 0 and this->get_time() <= n_load_steps)
      {
        return G;
      }
    else
      {
        return dealii::unit_symmetric_tensor<dim, Number>();
      }
  }

private:
  dealii::Tensor<2, dim, Number> G;
  const unsigned int             n_load_steps;
};

int
main(int argc, char *argv[])
{
  std::cout << "Classical HCMM!" << std::endl;

  using namespace dealii;
  using namespace Mixture;
  using namespace Common;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPI_Comm                         mpi_communicator(MPI_COMM_WORLD);

  const unsigned int dim = 3;
  // set types to use
  using Number     = double;
  using VectorType = TrilinosWrappers::MPI::Vector;
  using MatrixType = TrilinosWrappers::SparseMatrix;

  //// Mixture
  using Mixture = Mixture_GR<dim, VectorType, MatrixType, Number>;

  //// PARAMETERS
  const std::string output_directory = "Famaey_cube_test/";
  const std::string filename = "Famaey_cube_test_DBC_single_field_modelE_caseU";
  // apply all the prestretches in the initial step
  const unsigned int n_load_steps = 1;
  //// IMPORTANT!!! ////
  // For this test case to work the constituent_fiber.h/.cpp has to be modified
  // to NOT divide by sigma_h since sigma_h in this case is 0! (homeostatic
  // prestretch is 1.0)

  // HCMM parameters
  const unsigned int fe_degree   = 1;
  const unsigned int quad_degree = fe_degree + 1;
  const bool         active_GR   = true;

  // geometry
  const unsigned int n_refinements = 0;
  // time
  const unsigned int n_time_steps = 101;
  const double       dt           = 1.0;

  // material parameters Maes et al., 2023
  const std::vector<Number> mass_fractions{0.8, 0.05, 0.05, 0.05, 0.05};

  // ELASTIN
  // need to scale by a factor of 2 because of constitutive equation
  const double E_elastin                = 2.0 * 0.0305 / 0.8;
  const double axial_prestretch_elastin = 1.2;
  const double kappa                    = 10.0 * E_elastin;

  //  COL
  const double k_1_col        = 2.0 * 0.0289 / 0.05;
  const double k_2_col        = 1.23;
  const double T_col          = 101.0;       // decay time
  const double k_col          = 0.1 / T_col; // gain parameter
  const double prestretch_col = 1.1;
  const double alpha          = M_PI / 8.0; // fiber orientation

  try
    {
      //// create triangulation
      auto triangulation =
        std::make_shared<parallel::distributed::Triangulation<dim>>(
          mpi_communicator);
      GridGenerator::hyper_cube(*triangulation, 0., 1., true);

      // ... or do global refinement
      triangulation->refine_global(n_refinements);

      //// MIXTURE
      // create GrowthStrategy
      auto growth_strategy = std::make_unique<
        GrowthStrategies::GrowthStrategyAnisotropic<dim, Number>>(
        dealii::Tensor<1, dim>({1., 0., 0.}));
      // create PrestretchStrategy
      auto prestretch_strategy = std::make_unique<
        PrestretchStrategies::PrestretchStrategyNone<dim, Number>>();

      // create constituent factories
      std::vector<
        std::unique_ptr<Constituents::ConstituentFactoryBase<dim, Number>>>
        constituent_factories;

      // Elastin prestretch
      Tensor<2, dim, Number> prestretch_elastin;
      const Number g_zz = std::pow(axial_prestretch_elastin, 1. / n_load_steps);

      prestretch_elastin[0][0] = 1.0 / std::sqrt(g_zz);
      prestretch_elastin[1][1] = 1.0 / std::sqrt(g_zz);
      prestretch_elastin[2][2] = g_zz;

      ElastinPrestretch<dim, Number> prestretch_function_elastin(
        prestretch_elastin, n_load_steps);

      // create constant prestretch function for fiber constituent
      auto prestretch_function_col =
        Functions::ConstantFunction<dim>(prestretch_col);

      // mixture density and initial mass fractions
      auto mixture_density =
        std::make_shared<Functions::ConstantFunction<dim>>(1.0);
      auto initial_mass_fractions =
        std::make_shared<Functions::ConstantFunction<dim>>(mass_fractions);

      // create factories
      // 1) a hyperelastic constituent
      const unsigned int elastin_id = 0;
      constituent_factories.push_back(
        std::make_unique<
          Constituents::ConstituentHyperelasticFactory<dim, Number>>(
          elastin_id,
          prestretch_function_elastin,
          [E_elastin, kappa](const Point<dim, Number> &point) {
            (void)point;

            // compressible NeoHooke
            return std::make_unique<
              Materials::NeoHooke_compressible<dim, Number>>(E_elastin, kappa);
          }));

      // 2) a fiber based constituent - y-direction
      const unsigned int col_id = 1;
      constituent_factories.push_back(
        std::make_unique<Constituents::ConstituentFiberFactory<dim, Number>>(
          col_id,
          prestretch_function_col,
          [k_1_col, k_2_col](const Point<dim, Number> &point) {
            (void)point;
            const auto fiber_orientation =
              dealii::Tensor<1, dim, Number>({0.0, 1.0, 0.0});

            return std::make_unique<Materials::FiberMaterial_1D<dim, Number>>(
              k_1_col, k_2_col, fiber_orientation);
          },
          dt,
          T_col,
          k_col));

      // 2) a fiber based constituent - z-direction
      const unsigned int col_id_2 = 2;
      constituent_factories.push_back(
        std::make_unique<Constituents::ConstituentFiberFactory<dim, Number>>(
          col_id_2,
          prestretch_function_col,
          [k_1_col, k_2_col](const Point<dim, Number> &point) {
            (void)point;
            const auto fiber_orientation =
              dealii::Tensor<1, dim, Number>({0.0, 0.0, 1.0});

            return std::make_unique<Materials::FiberMaterial_1D<dim, Number>>(
              k_1_col, k_2_col, fiber_orientation);
          },
          dt,
          T_col,
          k_col));

      // 3) a fiber based constituent - yz-direction 1
      const unsigned int col_id_3 = 3;
      constituent_factories.push_back(
        std::make_unique<Constituents::ConstituentFiberFactory<dim, Number>>(
          col_id_3,
          prestretch_function_col,
          [k_1_col, k_2_col, alpha](const Point<dim, Number> &point) {
            (void)point;
            const auto fiber_orientation = dealii::Tensor<1, dim, Number>(
              {0.0, std::cos(alpha), std::sin(alpha)});

            return std::make_unique<Materials::FiberMaterial_1D<dim, Number>>(
              k_1_col, k_2_col, fiber_orientation);
          },
          dt,
          T_col,
          k_col));

      // 3) a fiber based constituent - yz-direction 2
      const unsigned int col_id_4 = 4;
      constituent_factories.push_back(
        std::make_unique<Constituents::ConstituentFiberFactory<dim, Number>>(
          col_id_4,
          prestretch_function_col,
          [k_1_col, k_2_col, alpha](const Point<dim, Number> &point) {
            (void)point;
            const auto fiber_orientation = dealii::Tensor<1, dim, Number>(
              {0.0, std::cos(alpha), -std::sin(alpha)});

            return std::make_unique<Materials::FiberMaterial_1D<dim, Number>>(
              k_1_col, k_2_col, fiber_orientation);
          },
          dt,
          T_col,
          k_col));

      // create boundary descriptor for mixture
      BoundaryDescriptor<dim, Number> boundary_descriptor_HCMM;
      //// Dirichlet BCs
      boundary_descriptor_HCMM.add_dirichlet_bc(
        1,
        std::make_shared<Functions::ConstantFunction<dim>>(
          std::vector<Number>{0.0, 0.0, 0.0}),
        ComponentMask(std::vector<bool>{true, false, false}));
      boundary_descriptor_HCMM.add_dirichlet_bc(
        2,
        std::make_shared<Functions::ConstantFunction<dim>>(
          std::vector<Number>{0.0, 0.0, 0.0}),
        ComponentMask(std::vector<bool>{false, true, false}));
      boundary_descriptor_HCMM.add_dirichlet_bc(
        4,
        std::make_shared<Functions::ConstantFunction<dim>>(
          std::vector<Number>{0.0, 0.0, 0.0}),
        ComponentMask(std::vector<bool>{false, false, true}));
      boundary_descriptor_HCMM.add_dirichlet_bc(
        5,
        std::make_shared<Functions::ConstantFunction<dim>>(
          std::vector<Number>{0.0, 0.0, 0.0}),
        ComponentMask(std::vector<bool>{false, false, true}));
      // STRETCH in y-direction
      boundary_descriptor_HCMM.add_dirichlet_bc(
        3,
        std::make_shared<BoundaryConditionLinearRamp<dim>>(
          0.0, 1.0, std::vector<Number>{0.0, 0.5, 0.0}),
        ComponentMask(std::vector<bool>{false, true, false}));


      // create Mixture
      Mixture mixture(*triangulation,
                      fe_degree,
                      quad_degree,
                      std::move(growth_strategy),
                      std::move(prestretch_strategy),
                      std::move(constituent_factories),
                      boundary_descriptor_HCMM,
                      false);

      // setup mixture system
      mixture.setup_system();

      // setup mixture quadrature data
      // CAREFUL: initial_mass_fractions needs to have as many components as
      // there are constituents!!!
      mixture.setup_qp_data(mixture_density, initial_mass_fractions);

      //// set output quantities of mixture postprocessor
      auto &mixture_postprocessor = mixture.get_postprocessor();
      mixture_postprocessor.set_output_file_name(filename, output_directory);

      // add mixture level quantities to postprocess
      mixture_postprocessor.add_quantity_extractor_tensor(
        "E_GL_strain", Postprocessing::GLStrain_v2<dim, Number>());
      mixture_postprocessor.add_quantity_extractor_tensor(
        "PK2_stress_mixture", Postprocessing::PK2Stress<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_tensor(
        "Cauchy_stress_mixture", Postprocessing::CauchyStress<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_tensor(
        "F_mixture", Postprocessing::DeformationGradient<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_tensor(
        "PK1_stress_mixture", Postprocessing::PK1Stress<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_tensor(
        "PK2_stress_vol_mixture",
        Postprocessing::volumetric_PK2Stress<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar(
        "detF_mixture", Postprocessing::detF_mixture<dim, Number>);


      //// constituent level
      // constituent 1
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"mass_fraction_ratio_FBS", col_id}},
        Postprocessing::CurrentMassFractionRatio<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_tensor_constituent(
        {{"PK2_stress_FBS", col_id}},
        Postprocessing::PK2StressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"sigma_FBS", col_id}},
        Postprocessing::FiberCauchyStressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"sigma_h_FBS", col_id}},
        Postprocessing::HomeostaticCauchyStressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_vector_constituent(
        {{"fiber_orientation_FBS", col_id}},
        Postprocessing::FiberOrientation<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"fbs_lambda_r", col_id}},
        Postprocessing::CurrentLambdaR<dim, Number>);
      // constituent 2
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"mass_fraction_ratio_col_2", col_id_2}},
        Postprocessing::CurrentMassFractionRatio<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_tensor_constituent(
        {{"PK2_stress_col_2", col_id_2}},
        Postprocessing::PK2StressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"sigma_col_2", col_id_2}},
        Postprocessing::FiberCauchyStressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"sigma_h_col_2", col_id_2}},
        Postprocessing::HomeostaticCauchyStressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_vector_constituent(
        {{"fiber_orientation_col_2", col_id_2}},
        Postprocessing::FiberOrientation<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"lambda_r_col_2", col_id_2}},
        Postprocessing::CurrentLambdaR<dim, Number>);
      // constituent 2
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"mass_fraction_ratio_col_3", col_id_3}},
        Postprocessing::CurrentMassFractionRatio<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_tensor_constituent(
        {{"PK2_stress_col_3", col_id_3}},
        Postprocessing::PK2StressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"sigma_col_3", col_id_3}},
        Postprocessing::FiberCauchyStressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"sigma_h_col_3", col_id_3}},
        Postprocessing::HomeostaticCauchyStressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_vector_constituent(
        {{"fiber_orientation_col_3", col_id_3}},
        Postprocessing::FiberOrientation<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"lambda_r_col_3", col_id_3}},
        Postprocessing::CurrentLambdaR<dim, Number>);
      // constituent 2
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"mass_fraction_ratio_col_4", col_id_4}},
        Postprocessing::CurrentMassFractionRatio<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_tensor_constituent(
        {{"PK2_stress_col_4", col_id_4}},
        Postprocessing::PK2StressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"sigma_col_4", col_id_4}},
        Postprocessing::FiberCauchyStressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"sigma_h_col_4", col_id_4}},
        Postprocessing::HomeostaticCauchyStressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_vector_constituent(
        {{"fiber_orientation_col_4", col_id_4}},
        Postprocessing::FiberOrientation<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"lambda_r_col_4", col_id_4}},
        Postprocessing::CurrentLambdaR<dim, Number>);
      // constituent 3
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"mass_fraction_ratio_elastin", elastin_id}},
        Postprocessing::CurrentMassFractionRatio<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_tensor_constituent(
        {{"PK2_stress_elastin", elastin_id}},
        Postprocessing::PK2StressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_tensor_constituent(
        {{"prestretch", elastin_id}},
        Postprocessing::PrestretchTensorConstituent<dim, Number>);

      //// NOX solver settings
      typename dealii::TrilinosWrappers::NOXSolver<VectorType>::AdditionalData
        nox_additional_data;
      nox_additional_data.abs_tol  = 1.e-12;
      nox_additional_data.rel_tol  = 1.e-12;
      nox_additional_data.max_iter = 15;
      Teuchos::RCP<Teuchos::ParameterList> non_linear_parameters =
        Teuchos::rcp(new Teuchos::ParameterList);
      auto &printParams = non_linear_parameters->sublist("Printing");
      printParams.set("Output Information", 0);
      mixture.setup_NOX(nox_additional_data, non_linear_parameters);

      // initial output at time 0
      mixture.output_results(0);

      //// RUN SIMULATION
      // solve steps and write output
      for (unsigned int step = 1; step <= n_time_steps; ++step)
        {
          // solve step using NOX
          mixture.solve_step_with_NOX(step, step * dt, active_GR);
          // write output
          mixture.output_results(step);
        }

      mixture.print_timer_stats();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}