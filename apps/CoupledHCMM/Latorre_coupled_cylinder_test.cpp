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
#include <MSHCMM/models/fully_coupled_model_arkode.h>
#include <MSHCMM/pathways/equations/pathway_Irons2020.h>
#include <MSHCMM/utilities/scalers.h>

#include <iostream>
#include <memory>

int
main(int argc, char *argv[])
{
  std::cout << "Coupled HCMM!" << std::endl;

  using namespace dealii;

  using namespace Models;
  using namespace Pathways;
  using namespace Mixture;
  using namespace Diffusion;
  using namespace Common;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPI_Comm                         mpi_communicator(MPI_COMM_WORLD);

  const unsigned int dim = 3;
  // set types to use
  using Number     = double;
  using VectorType = TrilinosWrappers::MPI::Vector;
  using MatrixType = TrilinosWrappers::SparseMatrix;

  //// Mixture
  using Mixture   = Mixture_GR<dim, VectorType, MatrixType, Number>;
  using Diffusion = DiffusionManager<dim, VectorType, MatrixType, Number>;
  using Coupled = FullyCoupledModelARKode<dim, VectorType, MatrixType, Number>;

  //// PARAMETERS
  const std::string output_directory = "Latorre_cylinder/";
  const std::string filename =
    "Latorre_coupled_cylinder_test_quadratic_s_stress_1_dP_10p";
  // prestretch
  const std::string prestretched_configuration =
    "../Prestretching/Latorre_cylinder/Latorre_cylinder_test_prestretch_quadratic.mesh";

  // HCMM parameters
  const unsigned int fe_degree   = 2;
  const unsigned int quad_degree = fe_degree + 1;

  // simulation
  const unsigned int n_time_steps = 150;
  const double       initial_time = 0.0;
  const double       final_time   = n_time_steps * 7.0;
  const double       rabstol =
    1.e-8; // residual absolute tolerance of time stepper in diffusion
  const double max_dt =
    (final_time - initial_time) / static_cast<double>(n_time_steps);

  // HCMM parameters
  const unsigned int solve_HCMM_every_x_steps = 1;
  const bool         active_GR                = true;
  const bool         coupled                  = true;
  // compute time step size of the mixture problem, multiple of the diffusion
  // problem time step size can be equal if HCMM problem is solved every
  // (diffusion) step
  const double time_step_size_HCMM =
    solve_HCMM_every_x_steps * (final_time - initial_time) / n_time_steps;
  // pathway parameters
  const bool         use_single_cell_collection    = false;
  const unsigned int solve_pathways_every_x_steps  = 1;
  const unsigned int output_pathways_every_x_steps = 1;
  const double       equilibration_time            = 100.0;
  // diffusion parameters
  const unsigned int output_diffusion_every_x_steps = 1;
  // cylinder 1
  const std::vector<double> initial_condition_diffusion = {0.34, 0.33, 0.33};

  const std::vector<std::string> diffusion_component_names = {"elastin",
                                                              "collagen",
                                                              "SMC"};
  const std::vector<Number>      D_coefficients{0.0, 0.0, 0.0};

  // degradation parameters
  const double T_col = 70.0;
  const double k_col = 1. / (T_col);
  const double T_smc = 70.0;
  const double k_smc = 1. / (T_smc);


  //// material parameters from Latorre et al., 2020, also Irons et al., 2021
  /// Table 2 (mouse descending thoracic aorta)
  // parameters homogenized to single-layered model
  const double mixture_density_value = 1.0;
  const double E_elastin             = 89.71;
  const double kappa                 = 10.0 * E_elastin;
  const double k_1_col               = 234.9;
  const double k_2_col               = 4.08;
  const double col_prestretch        = 1.25;
  const double k_1_smc               = 261.4;
  const double k_2_smc               = 0.24;
  const double smc_prestretch        = 1.2;
  const double angle = 0.52202798; // 29.91° wrt tube axis, in radians
  const std::vector<double> relative_collagen_contributions = {0.056,
                                                               0.067,
                                                               0.4385,
                                                               0.4385};
  const std::vector<double> mass_fractions                  = {
    0.34,                                      // elastin
    0.33 * relative_collagen_contributions[0], // col circ
    0.33 * relative_collagen_contributions[1], // col axial
    0.33 * relative_collagen_contributions[2], // col helix 1
    0.33 * relative_collagen_contributions[3], // col helix 2
    0.33};                                     // SMC (circ)
  // geometry data
  const double pressure       = 1.10 * 14.0;
  const double L              = 0.04;
  const double inner_radius   = 0.647;
  const double wall_thickness = 0.04;

  // pathway settings
  const double sensitivity_stress = 0.5;
  const double y_0_stress         = 0.2;
  const double y_0_AngII          = 0.0;
  const double y_0_WSS            = 0.5;
  const double y_0_SACs           = 0.2;
  const double y_0_integrins      = 0.2;

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
        Functions::ConstantFunction<dim>(smc_prestretch);
      auto prestretch_function_col =
        Functions::ConstantFunction<dim>(col_prestretch);

      // mixture density and initial mass fractions
      auto mixture_density = std::make_shared<Functions::ConstantFunction<dim>>(
        mixture_density_value);
      auto initial_mass_fractions_mixture =
        std::make_shared<Functions::ConstantFunction<dim>>(mass_fractions);
      // relative Collagen contributions
      auto relative_collagen_contributions_mixture =
        std::make_shared<Functions::ConstantFunction<dim>>(
          relative_collagen_contributions);
      // initial conditions DiffusionManager
      auto initial_mass_fractions_diffusion =
        std::make_shared<Functions::ConstantFunction<dim>>(
          initial_condition_diffusion);


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
          M_PI_2 - angle);
      // for fiber orientation of fiber 4 - helix 1
      const auto cos_transformer_4 =
        Constituents::CylindricalCoordinateTransformer<dim>(
          triangulation->get_manifold(0),
          inner_radius,
          outer_radius,
          M_PI_2 + angle);

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
          time_step_size_HCMM,
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
          time_step_size_HCMM,
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
          time_step_size_HCMM,
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
          time_step_size_HCMM,
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
          time_step_size_HCMM,
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
      /// constant pressure
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
                      coupled);

      // setup mixture system
      mixture.setup_system();

      // setup mixture quadrature data
      // CAREFUL: initial_mass_fractions needs to have as many components as
      // there are constituents!!!
      mixture.setup_qp_data(mixture_density, initial_mass_fractions_mixture);

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

      //// TRANSFORMERS - add transformers from pathways and diffusion to HCMM
      // Circ collagen: constituent that is produced by constituent 5 (SMCs)
      // NOTE: lambda needs to be mutable to that initial_mass_fractions can be
      // "modified", i.e., filled with values by initial_mass_fraction_func
      auto HCMM_col_circ_transformer =
        [k_col,
         D_coefficients,
         initial_mass_fractions_diffusion,
         initial_mass_fractions =
           Vector<Number>(initial_mass_fractions_diffusion->n_components),
         relative_collagen_contributions_mixture,
         relative_col_id =
           0](const dealii::Point<dim>  &p,
              const Number               time,
              const Vector<Number>      &diffusion_values,
              const Vector<Number>      &diffusion_laplacians,
              const std::vector<Number> &average_pathway_output,
              const std::vector<Number> &average_baseline_pathway_output,
              Constituents::TransferableParameters<Number>
                &transferable_parameters) mutable -> void {
        // not needed
        (void)time;
        // current density of the constituent
        // NOTE: diffusion_values[0] is NOT equal to the growth scalar!!! The
        // growth scalar actually is diffusion_values[0] /
        // initial_mass_fraction[]. This is automatically corrected in the
        // Mixture module if it is set up with coupled = true! Otherwise, the
        // "classical", uncoupled version is used which does not care about that
        // since it doesn't need transformers.

        // diffusion_values[1] corresponds to current mass fraction of TOTAL
        // collagen. This needs to be scaled by the relative contribution of the
        // collagen family associated with this transformer.
        // TODO: double check if the current mass fraction is then appropriately
        // scaled by the
        // RELATIVE contribution of this constituent in the mixture.

        const double relative_collagen_fraction =
          relative_collagen_contributions_mixture->value(p, relative_col_id);

        transferable_parameters.current_mass_fraction =
          diffusion_values[1] * relative_collagen_fraction;
        // compute normalized change in pathway output "collagen"
        Number delta_psi_col =
          ((average_pathway_output[0] + average_pathway_output[1]) -
           (average_baseline_pathway_output[0] +
            average_baseline_pathway_output[1])) /
          (average_baseline_pathway_output[0] +
           average_baseline_pathway_output[1]);
        // need scaling factor/mass fraction ratio for collagen mass fraction
        // function at the quadrature point.
        initial_mass_fractions_diffusion->vector_value(p,
                                                       initial_mass_fractions);

        // update mass production rate of constituent based on pathway output
        // "C" NOTE: the mass production term needs to be exactly the same as in
        // the diffusion transformer todo: IMPORTANT to use += in case there are
        // several cells producing the same constituent e.g.: SMCs and FBs both
        // can produce collagen since in this case we only have one cell type
        // producing the collagen, it is not necessary (doesnt hurt though)
        transferable_parameters.mass_production =
          (k_col * initial_mass_fractions[1] / initial_mass_fractions[2] *
             (1.0 + delta_psi_col) * diffusion_values[2] +
           D_coefficients[1] * diffusion_laplacians[1]) *
          relative_collagen_fraction;
      };
      mixture.add_transformer(col_id,
                              Cells::CellType::SMC,
                              HCMM_col_circ_transformer);

      /// constituent 2
      // Axial collagen: constituent that is produced by constituent 5 (SMCs)
      auto HCMM_col_axial_transformer =
        [k_col,
         D_coefficients,
         initial_mass_fractions_diffusion,
         initial_mass_fractions =
           Vector<Number>(initial_mass_fractions_diffusion->n_components),
         relative_collagen_contributions_mixture,
         relative_col_id =
           1](const dealii::Point<dim>  &p,
              const Number               time,
              const Vector<Number>      &diffusion_values,
              const Vector<Number>      &diffusion_laplacians,
              const std::vector<Number> &average_pathway_output,
              const std::vector<Number> &average_baseline_pathway_output,
              Constituents::TransferableParameters<Number>
                &transferable_parameters) mutable -> void {
        // not needed
        (void)time;

        // diffusion_values[1] corresponds to current mass fraction of TOTAL
        // collagen. This needs to be scaled by the relative contribution of the
        // collagen family associated with this transformer.
        // TODO: double check if the current mass fraction is then appropriately
        // scaled by the
        // RELATIVE contribution of this constituent in the mixture.
        const double relative_collagen_fraction =
          relative_collagen_contributions_mixture->value(p, relative_col_id);

        transferable_parameters.current_mass_fraction =
          diffusion_values[1] * relative_collagen_fraction;
        // compute normalized change in pathway output "collagen"
        Number delta_psi_col =
          ((average_pathway_output[0] + average_pathway_output[1]) -
           (average_baseline_pathway_output[0] +
            average_baseline_pathway_output[1])) /
          (average_baseline_pathway_output[0] +
           average_baseline_pathway_output[1]);

        // need scaling factor/mass fraction ratio for collagen mass fraction
        // function at the quadrature point.
        initial_mass_fractions_diffusion->vector_value(p,
                                                       initial_mass_fractions);

        // update mass production rate of constituent based on pathway output
        // needs to be exactly the same as in the diffusion transformer except
        // for additional scaling factor relative_collagen_fraction
        transferable_parameters.mass_production =
          (k_col * initial_mass_fractions[1] / initial_mass_fractions[2] *
             (1.0 + delta_psi_col) * diffusion_values[2] +
           D_coefficients[1] * diffusion_laplacians[1]) *
          relative_collagen_fraction;
      };

      // test new transformer (constituent_id, CellType, Transformer)
      mixture.add_transformer(col_id_2,
                              Cells::CellType::SMC,
                              HCMM_col_axial_transformer);

      /// constituent 3
      // Helix 1 collagen: constituent that is produced by constituent 5 (SMCs)
      auto HCMM_col_helix_1_transformer =
        [k_col,
         D_coefficients,
         initial_mass_fractions_diffusion,
         initial_mass_fractions =
           Vector<Number>(initial_mass_fractions_diffusion->n_components),
         relative_collagen_contributions_mixture,
         relative_col_id =
           2](const dealii::Point<dim>  &p,
              const Number               time,
              const Vector<Number>      &diffusion_values,
              const Vector<Number>      &diffusion_laplacians,
              const std::vector<Number> &average_pathway_output,
              const std::vector<Number> &average_baseline_pathway_output,
              Constituents::TransferableParameters<Number>
                &transferable_parameters) mutable -> void {
        // not needed
        (void)time;

        // diffusion_values[1] corresponds to current mass fraction of TOTAL
        // collagen. This needs to be scaled by the relative contribution of the
        // collagen family associated with this transformer.
        // TODO: double check if the current mass fraction is then appropriately
        // scaled by the
        // RELATIVE contribution of this constituent in the mixture.
        const double relative_collagen_fraction =
          relative_collagen_contributions_mixture->value(p, relative_col_id);

        transferable_parameters.current_mass_fraction =
          diffusion_values[1] * relative_collagen_fraction;
        // compute normalized change in pathway output "collagen"
        Number delta_psi_col =
          ((average_pathway_output[0] + average_pathway_output[1]) -
           (average_baseline_pathway_output[0] +
            average_baseline_pathway_output[1])) /
          (average_baseline_pathway_output[0] +
           average_baseline_pathway_output[1]);

        // need scaling factor/mass fraction ratio for collagen mass fraction
        // function at the quadrature point.
        initial_mass_fractions_diffusion->vector_value(p,
                                                       initial_mass_fractions);

        // update mass production rate of constituent based on pathway output
        transferable_parameters.mass_production =
          (k_col * initial_mass_fractions[1] / initial_mass_fractions[2] *
             (1.0 + delta_psi_col) * diffusion_values[2] +
           D_coefficients[1] * diffusion_laplacians[1]) *
          relative_collagen_fraction;
      };

      // test new transformer (constituent_id, CellType, Transformer)
      mixture.add_transformer(col_id_3,
                              Cells::CellType::SMC,
                              HCMM_col_helix_1_transformer);

      /// constituent 4
      // Helix 1 collagen: constituent that is produced by constituent 5 (SMCs)
      auto HCMM_col_helix_2_transformer =
        [k_col,
         D_coefficients,
         initial_mass_fractions_diffusion,
         initial_mass_fractions =
           Vector<Number>(initial_mass_fractions_diffusion->n_components),
         relative_collagen_contributions_mixture,
         relative_col_id =
           3](const dealii::Point<dim>  &p,
              const Number               time,
              const Vector<Number>      &diffusion_values,
              const Vector<Number>      &diffusion_laplacians,
              const std::vector<Number> &average_pathway_output,
              const std::vector<Number> &average_baseline_pathway_output,
              Constituents::TransferableParameters<Number>
                &transferable_parameters) mutable -> void {
        // not needed
        (void)time;

        // diffusion_values[1] corresponds to current mass fraction of TOTAL
        // collagen. This needs to be scaled by the relative contribution of the
        // collagen family associated with this transformer.
        // TODO: double check if the current mass fraction is then appropriately
        // scaled by the
        // RELATIVE contribution of this constituent in the mixture.

        // get relative collagen contribution of this collagen fiber family.
        // Note: since it is constant in space and time, the value could also
        // just be captured by the lambda function to avoid a function call
        const double relative_collagen_fraction =
          relative_collagen_contributions_mixture->value(p, relative_col_id);
        transferable_parameters.current_mass_fraction =
          diffusion_values[1] * relative_collagen_fraction;

        // compute normalized change in pathway output "collagen"
        Number delta_psi_col =
          ((average_pathway_output[0] + average_pathway_output[1]) -
           (average_baseline_pathway_output[0] +
            average_baseline_pathway_output[1])) /
          (average_baseline_pathway_output[0] +
           average_baseline_pathway_output[1]);

        // need scaling factor/mass fraction ratio for collagen mass fraction
        // function at the quadrature point. get total mass fractions, i.e., no
        // distinction between different collagen families is made
        initial_mass_fractions_diffusion->vector_value(p,
                                                       initial_mass_fractions);

        // update mass production rate of constituent based on pathway output
        // NOTE: the mass production term needs to be exactly the same as in the
        // diffusion transformer except for the relative_collagen_fraction
        // scaling factor
        transferable_parameters.mass_production =
          (k_col * initial_mass_fractions[1] / initial_mass_fractions[2] *
             (1.0 + delta_psi_col) * diffusion_values[2] +
           D_coefficients[1] * diffusion_laplacians[1]) *
          relative_collagen_fraction;
      };

      // test new transformer (constituent_id, CellType, Transformer)
      mixture.add_transformer(col_id_4,
                              Cells::CellType::SMC,
                              HCMM_col_helix_2_transformer);

      /// constituent 5
      // Circ SMCs: constituent that produces other constituents
      auto HCMM_smc_transformer =
        [k_smc, D_coefficients](
          const dealii::Point<dim>  &p,
          const Number               time,
          const Vector<Number>      &diffusion_values,
          const Vector<Number>      &diffusion_laplacians,
          const std::vector<Number> &average_pathway_output,
          const std::vector<Number> &average_baseline_pathway_output,
          Constituents::TransferableParameters<Number> &transferable_parameters)
        -> void {
        // not needed
        (void)p;
        (void)time;
        // current density of the constituent
        // NOTE: diffusion_values[0] is NOT equal to the growth scalar!!! The
        // growth scalar actually is diffusion_values[0] /
        // initial_mass_fraction[]. This is automatically corrected in the
        // Mixture module if it is set up with coupled = true! Otherwise, the
        // "classical", uncoupled version is used which does not care about that
        // since it doesn't need transformers.
        transferable_parameters.current_mass_fraction = diffusion_values[2];
        // compute normalized change in pathway output "proliferation"
        Number delta_psi_smc =
          (average_pathway_output[3] - average_baseline_pathway_output[3]) /
          average_baseline_pathway_output[3];
        // update mass production rate
        transferable_parameters.mass_production =
          k_smc * (1.0 + delta_psi_smc) * diffusion_values[2] +
          D_coefficients[2] * diffusion_laplacians[2];
      };

      // test new transformer (constituent_id, CellType, Transformer)
      mixture.add_transformer(smc_id,
                              Cells::CellType::SMC,
                              HCMM_smc_transformer);


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
      mixture_postprocessor.add_quantity_extractor_tensor(
        "Rotated_PK1Stress_mixture",
        Postprocessing::RotatedPK1Stress<dim, Number>(cos_transformer));
      mixture_postprocessor.add_quantity_extractor_tensor(
        "Rotated_F_mixture",
        Postprocessing::RotatedDefGrad<dim, Number>(cos_transformer));
      mixture_postprocessor.add_quantity_extractor_tensor(
        "Rotated_PK2_stress_mixture",
        Postprocessing::RotatedPK2Stress<dim, Number>(cos_transformer));

      // write scalar data from the mixture
      // these should be the correct versions
      mixture_postprocessor.add_quantity_extractor_scalar(
        "detF_mixture", Postprocessing::detF_mixture<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar(
        "trace_sigma", Postprocessing::trace_sigma_mixture<dim, Number>);


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
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"active_sigma_smc", smc_id}},
        Postprocessing::ActiveFiberCauchyStressConstituent<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_vector_constituent(
        {{"fiber_orientation_smc", smc_id}},
        Postprocessing::FiberOrientation<dim, Number>);
      mixture_postprocessor.add_quantity_extractor_scalar_constituent(
        {{"lambda_r_smc", smc_id}},
        Postprocessing::CurrentLambdaR<dim, Number>);


      //// DIFFUSION
      // initial conditions of the diffusion problem
      // contains all the constituents, i.e., elastin, collagen, and SMCs
      // Already created above since it is needed for HCMM transformers

      // create source term
      auto source_term = nullptr;

      // create diffusion problem
      Diffusion diffusion(*triangulation,
                          fe_degree,
                          quad_degree,
                          initial_mass_fractions_diffusion->n_components,
                          initial_mass_fractions_diffusion,
                          source_term);
      // set output filename and component names
      diffusion.set_output_filename(filename + "_diffusion", output_directory);
      diffusion.set_component_names(diffusion_component_names);

      //// create diffusion parameters
      const auto n_components = initial_mass_fractions_diffusion->n_components;
      // ZERO diffusion coefficients -> no diffusion
      auto diffusion_coefficients =
        std::make_shared<Functions::ConstantFunction<dim>>(D_coefficients);

      // linear terms
      std::vector<std::vector<Number>> K(n_components,
                                         std::vector<Number>(n_components,
                                                             0.0));
      // higher order terms
      HigherOrderMatrix<Number> Q(n_components);
      for (auto &ele : Q)
        {
          ele.resize(n_components, [](const dealii::Vector<Number> &y) {
            (void)y;
            return 0.0;
          });
        }

      // derivatives of higher order terms
      HigherOrderMatrix<Number> Q_derivative(n_components);
      for (auto &ele : Q_derivative)
        {
          ele.resize(n_components, [](const dealii::Vector<Number> &y) {
            (void)y;
            return 0.0;
          });
        }

      // setup system (includes interpolating initial conditions and creation of
      // initial LocalDiffusionParameters)
      diffusion.setup_system(diffusion_coefficients, K, Q, Q_derivative, true);


      //// DIFFUSION NEW TRANSFORMERS
      auto diffusion_transformer =
        [k_smc,
         k_col,
         initial_mass_fractions_diffusion,
         mass_fractions =
           Vector<Number>(initial_mass_fractions_diffusion->n_components)](
          const dealii::Point<dim>         &p,
          const Number                      time,
          const Vector<Number>             &diffusion_values,
          const std::vector<Number>        &average_pathway_output,
          const std::vector<Number>        &average_baseline_pathway_output,
          LocalDiffusionParameters<Number> &local_diffusion_parameters) mutable
        -> void {
        // not needed
        (void)time;
        (void)diffusion_values;
        // get output and baseline output of pathway of the cell type associated
        // with this transformer compute normalized difference from baseline
        // output "collagen"
        Number delta_psi_col =
          ((average_pathway_output[0] + average_pathway_output[1]) -
           (average_baseline_pathway_output[0] +
            average_baseline_pathway_output[1])) /
          (average_baseline_pathway_output[0] +
           average_baseline_pathway_output[1]);
        // output "proliferation"
        Number delta_psi_smc =
          (average_pathway_output[3] - average_baseline_pathway_output[3]) /
          average_baseline_pathway_output[3];
        // need scaling factor/mass fraction ratio for collagen mass fraction
        // function at the quadrature point which is stored in
        // local_cell_collection
        initial_mass_fractions_diffusion->vector_value(p, mass_fractions);

        // get linear matrix
        auto &linear_matrix = local_diffusion_parameters.get_K();
        // net mass production rate, first term is production, second term is
        // removal SMC mass fraction
        linear_matrix[2][2] = (k_smc * (1.0 + delta_psi_smc) - k_smc);
        // collagen mass fraction
        // constant degradation
        linear_matrix[1][1] = -k_col;
        // production proportional to SMC fraction, factor of mass_fractions[1]
        // / mass_fractions[0] ensures homeostasis in initial configuration
        // todo: double check if the indices are correct!
        linear_matrix[1][2] =
          mass_fractions[1] / mass_fractions[2] * k_col * (1.0 + delta_psi_col);
      };

      diffusion.add_transformer(Cells::CellType::SMC, diffusion_transformer);



      //// PATHWAYS
      // some settings for pathway solver
      typename dealii::SUNDIALS::ARKode<dealii::Vector<Number>>::AdditionalData
        pathway_solver_data;
      pathway_solver_data.absolute_tolerance       = 1.e-10;
      pathway_solver_data.relative_tolerance       = 1.e-10;
      pathway_solver_data.mass_is_time_independent = true;

      // create PathwayManager
      PathwayManager<dim, Number> pathway_manager(pathway_solver_data,
                                                  mpi_communicator);
      // create PathwayStorage...
      PathwayStorage<dim, Number> pathwayStorage;
      // ... create vector for all pathway equations for specific CellType
      std::vector<std::shared_ptr<Equations::PathwayEquationBase<Number>>>
        pathwayEquations;
      // NOTE: number of components in the pathway must match the defined
      // equation in the concrete class! pathways from Irons et al., 2020
      pathwayEquations.push_back(
        std::make_shared<Equations::PathwayIrons2020<Number>>());
      const std::vector<Number> pathway_weights{1.};

      // create transformer for pathway input (stress and diffusion)
      // if we want to have different scalers for different pathways of the same
      // cell type, we can use the passed pathway_id of the passed cell_state to
      // the function. Then we can use that to select the appropriate scaler for
      // that pathway which the lambda should capture. using different
      // sensitivities for different subpopulations
      const auto exp_scaler =
        Scalers::NEW_exponential(sensitivity_stress, y_0_stress);
      const unsigned int pathway_index_of_stress = 0;

      auto pathway_transformer =
        [exp_scaler, &y_0_AngII, &y_0_WSS, &y_0_SACs, &y_0_integrins](
          const dealii::Point<dim>     &p,
          const Number                  time,
          const dealii::Vector<Number> &diffusion_values,
          const Constituents::TransferableParameters<Number>
                                               &transferable_parameters,
          const unsigned int                    pathway_id,
          const dealii::Tensor<2, dim, Number> &deformation_gradient,
          const dealii::SymmetricTensor<2, dim, Number> &PK2_stress,
          dealii::Vector<Number> &cell_state) mutable -> void {
        // not needed
        (void)p;
        //(void)time;
        // not needed since we don't have several pathway
        (void)pathway_id;
        (void)deformation_gradient;
        (void)PK2_stress;
        //// if time is 0.0, we are equilibrating and should use sigma_hom for
        /// that
        const auto fiber_cauchy_stress =
          time == 0.0 ? transferable_parameters.sigma_hom :
                        transferable_parameters.fiber_cauchy_stress;
        // compute normalized stress deviation from homeostatic
        const auto delta_sigma =
          (fiber_cauchy_stress - transferable_parameters.sigma_hom) /
          transferable_parameters.sigma_hom;
        // set cell state for stress input
        cell_state[pathway_index_of_stress] = exp_scaler(delta_sigma);

        // IRONS pathway has only varying stress input
        cell_state[1] = y_0_WSS;       // WSS
        cell_state[2] = y_0_AngII;     // AngII
        cell_state[3] = y_0_SACs;      // SACs
        cell_state[4] = y_0_integrins; // Integrins

        // diffusion part
        // pathway does not have diffusion input, nothing to do in this case...
        (void)diffusion_values;
      };

      pathwayStorage.add_pathways_and_transformers(Cells::CellType::SMC,
                                                   std::move(pathwayEquations),
                                                   pathway_weights,
                                                   std::move(
                                                     pathway_transformer),
                                                   smc_id);
      // setup pathway manager
      pathway_manager.setup(pathwayStorage,
                            *triangulation,
                            fe_degree,
                            quad_degree,
                            use_single_cell_collection);
      // finalize setup by distributing the cells
      pathway_manager.distribute_cells();
      // setup pathway output. Has to be called AFTER distribute cells!
      std::map<Cells::CellType, std::vector<unsigned int>> cells_for_vtu;
      cells_for_vtu.emplace(Cells::CellType::SMC, std::vector<unsigned int>{});
      pathway_manager.setup_pathway_output(filename + "_pathways",
                                           true,
                                           cells_for_vtu,
                                           output_directory);


      //// COUPLED PROBLEM
      // create coupled problem by combining mixture, diffusion, and pathway
      // manager IMPORTANT: all the sub-problems should already be set up
      Coupled coupled_problem(mixture, diffusion, pathway_manager);
      // setup time stepping
      typename SUNDIALS::ARKode<VectorType>::AdditionalData data;
      data.initial_time                          = initial_time;
      data.final_time                            = final_time;
      data.absolute_tolerance                    = 1.e-8;
      data.relative_tolerance                    = 1.e-7;
      data.mass_is_time_independent              = true;
      data.maximum_order                         = 3; // default is 5
      data.implicit_function_is_time_independent = true;
      data.implicit_function_is_linear           = true;
      // data.initial_step_size = 0.01;
      coupled_problem.setup_time_stepper(data, max_dt, rabstol);


      //// Run coupled problem
      // get mixture from coupled problem
      auto &mix = coupled_problem.get_mixture();
      // read prestretched configuration
      mix.read_prestretched_configuration(elastin_id);
      // run
      coupled_problem.run(n_time_steps,
                          initial_time,
                          final_time,
                          output_diffusion_every_x_steps,
                          equilibration_time,
                          solve_pathways_every_x_steps,
                          output_pathways_every_x_steps,
                          solve_HCMM_every_x_steps,
                          active_GR);
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