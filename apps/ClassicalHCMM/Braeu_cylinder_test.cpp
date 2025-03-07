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
#include <MSHCMM/mixture/growth_strategies/growth_strategy_anisotropic_radial.h>
#include <MSHCMM/mixture/growth_strategies/growth_strategy_penalty.h>
#include <MSHCMM/mixture/materials/fiber_material_1D.h>
#include <MSHCMM/mixture/materials/neo_hooke_compressible.h>
#include <MSHCMM/mixture/prestretch_strategies/prestretch_strategy_famaey.h>
#include <MSHCMM/mixture/prestretch_strategies/prestretch_strategy_mousavi.h>
#include <MSHCMM/mixture/prestretch_strategies/prestretch_strategy_none.h>

#include <iostream>
#include <memory>

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
  const std::string output_directory = "Braeu_cylinder/";
  const std::string filename = "Braeu_cylinder_test_quadratic_k_04_dP_15p";
  // prestretch
  const std::string prestretched_configuration =
    "../Prestretching/Braeu_cylinder/Braeu_cylinder_test_prestretch_quadratic.mesh";

  // HCMM parameters
  const unsigned int fe_degree   = 2;
  const unsigned int quad_degree = fe_degree + 1;
  const bool         active_GR   = true;

  // time
  const unsigned int n_time_steps = 250;  //
  const double       dt           = 10.1; // [days]

  // geometry
  const double inner_radius   = 10.0;
  const double wall_thickness = 1.41;
  const double L              = wall_thickness;

  // material parameters Braeu et al., 2017
  const double              pressure = 1.15 * 13.3; // Braeu
  const std::vector<Number> mass_fractions{
    0.23, 0.062, 0.062, 0.248, 0.248, 0.15};

  // ELASTIN
  const Number E_elastin = 75.6;
  const Number kappa     = 10.0 * E_elastin;

  // SMC
  // need to scale by a factor of 2 because of constitutive equation
  const Number k_1_smc        = 2.0 * 7.98;
  const Number k_2_smc        = 11.4;
  const Number prestretch_smc = 1.1;
  const Number T_smc          = 101.0;       // decay time
  const Number k_smc          = 0.1 / T_smc; // gain parameter

  // COL
  // need to scale by a factor of 2 because of constitutive equation
  const double k_1_col        = 2.0 * 559.4;
  const double k_2_col        = 11.2;
  const double prestretch_col = 1.062;
  const Number T_col          = 101.0;       // decay time
  const Number k_col          = 0.1 / T_col; // gain parameter
  const Number alpha          = M_PI / 4;

  try
    {
      //// create triangulation
      auto triangulation =
        std::make_shared<parallel::distributed::Triangulation<dim>>(
          mpi_communicator);

      // set dimensions of the triangulation
      const Point<dim - 1> center(0., 0.);
      const double         outer_radius = inner_radius + wall_thickness;
      // total number of cells in the shell, just one layer
      // if 0, then number of cells are computed adaptively such that they have
      // the smallest aspect ratio, see deal.ii documentation
      const unsigned int n_cells  = 0;
      const bool         colorize = true;
      // n_slices = number of cells in axial direction + 1
      const unsigned int n_slices = 2; // 2;


      // 1) create 2D triangulation
      Triangulation<dim - 1> tria_temp;
      // 2) create half a hypershell in 2D (note: also works with
      // quarter_hyper_shell)
      GridGenerator::quarter_hyper_shell(
        tria_temp, center, inner_radius, outer_radius, n_cells, colorize);
      // 3) extrude mesh in z-direction
      // 3.1) create final 3D triangulation
      Triangulation<dim> tria;
      GridGenerator::extrude_triangulation(tria_temp, n_slices, L, tria);
      // 3.2) attach cylindrical manifold to triangulation - IMPORTANT to get a
      // good shape
      tria.set_all_manifold_ids(0);
      const unsigned int axis = 2; // 2 - z-axis, direction of tube
      tria.set_manifold(0, CylindricalManifold<dim>(axis));
      // copy serial triangulation to distributed triangulation
      triangulation->copy_triangulation(tria);

      // read prestretched configuration...
      triangulation->load(prestretched_configuration);
      // ... or do global refinement
      // triangulation->refine_global(n_refinements);

      //// MIXTURE
      // Global Coordinate transformer from global cartesian COS to local
      // cylindrical COS and back used for prestretch of elastin and rotated
      // Cauchy stress
      const auto cos_transformer =
        Constituents::CylindricalCoordinateTransformer<dim>(
          triangulation->get_manifold(0), inner_radius, outer_radius, M_PI_4);
      // create GrowthStrategy
      auto growth_strategy = std::make_unique<
        GrowthStrategies::GrowthStrategyAnisotropicRadial<dim, Number>>();
      // create PrestretchStrategy
      auto prestretch_strategy = std::make_unique<
        PrestretchStrategies::PrestretchStrategyNone<dim, Number>>();

      // create constituent factories
      std::vector<
        std::unique_ptr<Constituents::ConstituentFactoryBase<dim, Number>>>
        constituent_factories;

      // create constant prestretch function for elastin
      ConstantTensorFunction<2, dim, Number> prestretch_function_elastin(
        unit_symmetric_tensor<dim>());

      // create constant prestretch function for fiber constituent
      auto prestretch_function_smc =
        Functions::ConstantFunction<dim>(prestretch_smc);
      auto prestretch_function_col =
        Functions::ConstantFunction<dim>(prestretch_col);

      // mixture density and initial mass fractions
      auto mixture_density =
        std::make_shared<Functions::ConstantFunction<dim>>(1.0);
      auto initial_mass_fractions =
        std::make_shared<Functions::ConstantFunction<dim>>(mass_fractions);

      // create cylindrical coordinate transformers
      // for fiber orientation of fiber 1 - circumferential
      // rotation angle of 0 -> fiber aligned in circumferential system
      // i.e., the rotation angle is relative to circumferential direction
      const auto cos_transformer_1 =
        Constituents::CylindricalCoordinateTransformer<dim>(
          triangulation->get_manifold(0), inner_radius, outer_radius, 0.0);
      // for fiber orientation of fiber 2 - axial
      const auto cos_transformer_2 =
        Constituents::CylindricalCoordinateTransformer<dim>(
          triangulation->get_manifold(0), inner_radius, outer_radius, M_PI_2);
      // for fiber orientation of fiber 3 - helix 1
      const auto cos_transformer_3 =
        Constituents::CylindricalCoordinateTransformer<dim>(
          triangulation->get_manifold(0),
          inner_radius,
          outer_radius,
          M_PI - alpha);
      // for fiber orientation of fiber 4 - helix 1
      const auto cos_transformer_4 =
        Constituents::CylindricalCoordinateTransformer<dim>(
          triangulation->get_manifold(0), inner_radius, outer_radius, alpha);

      //// create factories
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
          [k_1_col, k_2_col, cos_transformer_1](
            const Point<dim, Number> &point) {
            (void)point;
            const auto fiber_orientation = cos_transformer_1.get_tangent(point);

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
          [k_1_col, k_2_col, cos_transformer_2](
            const Point<dim, Number> &point) {
            (void)point;
            const auto fiber_orientation = cos_transformer_2.get_tangent(point);

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
          [k_1_col, k_2_col, cos_transformer_3](
            const Point<dim, Number> &point) {
            (void)point;
            const auto fiber_orientation = cos_transformer_3.get_tangent(point);

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
          [k_1_col, k_2_col, cos_transformer_4](
            const Point<dim, Number> &point) {
            (void)point;
            const auto fiber_orientation = cos_transformer_4.get_tangent(point);

            return std::make_unique<Materials::FiberMaterial_1D<dim, Number>>(
              k_1_col, k_2_col, fiber_orientation);
          },
          dt,
          T_col,
          k_col));

      // 2) a fiber based constituent - y-direction
      const unsigned int smc_id = 5;
      constituent_factories.push_back(
        std::make_unique<Constituents::ConstituentFiberFactory<dim, Number>>(
          smc_id,
          prestretch_function_smc,
          [k_1_smc, k_2_smc, cos_transformer_1](
            const Point<dim, Number> &point) {
            (void)point;
            const auto fiber_orientation = cos_transformer_1.get_tangent(point);

            return std::make_unique<Materials::FiberMaterial_1D<dim, Number>>(
              k_1_smc, k_2_smc, fiber_orientation);
          },
          dt,
          T_smc,
          k_smc));

      //// Dirichlet BCs
      // create boundary descriptor for mixture
      BoundaryDescriptor<dim, Number> boundary_descriptor_HCMM;
      // add dirichlet conditions: fix bottom in z-direction
      boundary_descriptor_HCMM.add_dirichlet_bc(
        4,
        std::make_shared<Functions::ConstantFunction<dim>>(
          std::vector<Number>{0.0, 0.0, 0.0}),
        ComponentMask(std::vector<bool>{false, false, true}));
      // add dirichlet conditions: fix top in z-direction
      boundary_descriptor_HCMM.add_dirichlet_bc(
        5,
        std::make_shared<Functions::ConstantFunction<dim>>(
          std::vector<Number>{0.0, 0.0, 0.0}),
        ComponentMask(std::vector<bool>{false, false, true}));
      // fix upper cut in x-direction
      boundary_descriptor_HCMM.add_dirichlet_bc(
        2,
        std::make_shared<Functions::ConstantFunction<dim>>(
          std::vector<Number>{0.0, 0.0, 0.0}),
        ComponentMask(std::vector<bool>{true, false, false}));
      // fix lower cut in x-direction
      boundary_descriptor_HCMM.add_dirichlet_bc(
        3,
        std::make_shared<Functions::ConstantFunction<dim>>(
          std::vector<Number>{0.0, 0.0, 0.0}),
        ComponentMask(std::vector<bool>{false, true, false}));
      //// Neumann BC
      // constant pressure
      boundary_descriptor_HCMM.add_pressure_bc(
        0, std::make_shared<Functions::ConstantFunction<dim>>(-pressure));

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
      mixture_postprocessor.add_quantity_extractor_tensor(
        "Rotated_CauchyStress_mixture",
        Postprocessing::RotatedCauchyStress<dim, Number>(cos_transformer));
      mixture_postprocessor.add_quantity_extractor_scalar(
        "detF_mixture", Postprocessing::detF_mixture<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar(
        "trace_sigma", Postprocessing::trace_sigma_mixture<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_tensor(
        "Rotated_PK1Stress_mixture",
        Postprocessing::RotatedPK1Stress<dim, Number>(cos_transformer));
      mixture_postprocessor.add_quantity_extractor_tensor(
        "Rotated_F_mixture",
        Postprocessing::RotatedDefGrad<dim, Number>(cos_transformer));
      mixture_postprocessor.add_quantity_extractor_tensor(
        "Rotated_PK2_stress_mixture",
        Postprocessing::RotatedPK2Stress<dim, Number>(cos_transformer));

      //// constituent level
      // constituent 1
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"mass_fraction_ratio_col_1", col_id}},
        Postprocessing::CurrentMassFractionRatio<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_tensor_constituent(
        {{"PK2_stress_col_1", col_id}},
        Postprocessing::PK2StressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"sigma_col_1", col_id}},
        Postprocessing::FiberCauchyStressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"sigma_h_col_1", col_id}},
        Postprocessing::HomeostaticCauchyStressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_vector_constituent(
        {{"fiber_orientation_col_1", col_id}},
        Postprocessing::FiberOrientation<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"lambda_r_col_1", col_id}},
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
      // constituent 3
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
      // constituent 4
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
      // constituent 5
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"mass_fraction_ratio_elastin", elastin_id}},
        Postprocessing::CurrentMassFractionRatio<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_tensor_constituent(
        {{"PK2_stress_elastin", elastin_id}},
        Postprocessing::PK2StressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_tensor_constituent(
        {{"prestretch", elastin_id}},
        Postprocessing::PrestretchTensorConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_tensor_constituent(
        {{"rotated_prestretch", elastin_id}},
        Postprocessing::RotatedPrestretch<dim, Number>(cos_transformer));
      // constituent 6
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"mass_fraction_ratio_smc", smc_id}},
        Postprocessing::CurrentMassFractionRatio<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_tensor_constituent(
        {{"PK2_stress_smc", smc_id}},
        Postprocessing::PK2StressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"sigma_smc", smc_id}},
        Postprocessing::FiberCauchyStressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"sigma_h_smc", smc_id}},
        Postprocessing::HomeostaticCauchyStressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_vector_constituent(
        {{"fiber_orientation_smc", smc_id}},
        Postprocessing::FiberOrientation<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"lambda_r_smc", smc_id}},
        Postprocessing::CurrentLambdaR<dim, Number>);

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

      /// READ PRESTRETCHED CONFIGURATION
      // read prestretched configuration
      mixture.read_prestretched_configuration(elastin_id);

      // initial output at time 0
      mixture.output_results(0);

      //// RUN SIMULATION
      // solve steps and write output
      for (unsigned int step = 1; step <= n_time_steps; ++step)
        {
          // solve step using NOX
          mixture.solve_step_with_NOX(step, /*time=*/step * dt, active_GR);
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