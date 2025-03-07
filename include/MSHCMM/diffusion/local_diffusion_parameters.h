#ifndef MSHCMM_LOCAL_DIFFUSION_PARAMETERS_H
#define MSHCMM_LOCAL_DIFFUSION_PARAMETERS_H

#include <vector>

namespace Diffusion
{
  //! type alias
  template <typename Number = double>
  using ConstantCoefficientMatrix = std::vector<std::vector<Number>>;

  //! type alias
  template <typename Number = double>
  using HigherOrderMatrix = std::vector<
    std::vector<std::function<Number(const dealii::Vector<Number> &)>>>;

  template <typename Number = double>
  class LocalDiffusionParameters
  {
  public:
    // need default constructor to use in CellDataStorage!
    LocalDiffusionParameters() = default;

    LocalDiffusionParameters(const unsigned int n_species,
                             const bool         is_constant)
      : n_species(n_species)
      , constant(is_constant)
    {
      // set size of parameters
      this->resize_parameters(n_species);
    }

    LocalDiffusionParameters(const std::vector<std::vector<Number>> &K,
                             const HigherOrderMatrix<Number>        &Q,
                             const HigherOrderMatrix<Number> &Q_derivative)
      : K(K)
      , Q(Q)
      , Q_derivative(Q_derivative)
    {}

    void
    resize_parameters(const unsigned int n_species_in)
    {
      n_species = n_species_in;

      // resize K
      K.resize(n_species_in);
      for (auto &k : K)
        k.resize(n_species_in, 0.0);

      // quadratic terms
      // adjust size of Q to n_components x n_components, initialize to function
      // which returns 0
      Q.resize(n_species_in);
      for (auto &ele : Q)
        {
          ele.resize(n_species_in, [](const dealii::Vector<Number> &y) {
            (void)y;
            return 0.0;
          });
        }
      // derivative of quadratic terms
      // adjust size of Q_derivative to n_components x n_components, initialize
      // to function which returns 0
      Q_derivative.resize(n_species_in);
      for (auto &ele : Q_derivative)
        {
          ele.resize(n_species_in, [](const dealii::Vector<Number> &y) {
            (void)y;
            return 0.0;
          });
        }
    }

    // set initial parameters
    // todo: add function to set the initial parameters
    void
    set_parameters()
    {
      // linear terms
      K[0][0] = 0.0; //-1.0;
      // K[1][1] = 0.0;//-1.0;

      // quadratic term terms
      // Q[0][0] = 1.0;
    }

    void
    set_linear_coefficients(
      const ConstantCoefficientMatrix<Number> &linear_coefficients)
    {
      Assert(
        linear_coefficients.size() == n_species and
          linear_coefficients[0].size() == n_species,
        dealii::ExcMessage(
          "Size of given matrix does not match the number of species set in local diffusion parameters!"));
      K = linear_coefficients;
    }

    void
    set_higher_order_coefficients(
      const HigherOrderMatrix<Number> &higher_order_coefficients,
      const HigherOrderMatrix<Number> &higher_order_coefficients_derivative)
    {
      Assert(
        higher_order_coefficients.size() == n_species and
          higher_order_coefficients[0].size() == n_species,
        dealii::ExcMessage(
          "Size of given matrix does not match the number of species set in local diffusion parameters!"));

      Assert(
        higher_order_coefficients_derivative.size() == n_species and
          higher_order_coefficients_derivative[0].size() == n_species,
        dealii::ExcMessage(
          "Size of given matrix does not match the number of species set in local diffusion parameters!"));

      Q            = higher_order_coefficients;
      Q_derivative = higher_order_coefficients_derivative;
    }

    bool
    is_constant()
    {
      return constant;
    }

    [[nodiscard]] unsigned int
    get_n_species() const
    {
      return n_species;
    }

    const ConstantCoefficientMatrix<Number> &
    get_K() const
    {
      return K;
    }

    ConstantCoefficientMatrix<Number> &
    get_K()
    {
      return K;
    }

    const HigherOrderMatrix<Number> &
    get_Q() const
    {
      return Q;
    }

    HigherOrderMatrix<Number> &
    get_Q()
    {
      return Q;
    }

    const HigherOrderMatrix<Number> &
    get_Q_derivative() const
    {
      return Q_derivative;
    }

    HigherOrderMatrix<Number> &
    get_Q_derivative()
    {
      return Q_derivative;
    }

    Number
    get_linear_coefficient(const unsigned int component_i,
                           const unsigned int component_j) const
    {
      return K[component_i][component_j];
    }

    Number
    get_quadratic_coefficient(const unsigned int            component_i,
                              const unsigned int            component_j,
                              const dealii::Vector<Number> &values) const
    {
      return Q[component_i][component_j](values);
    }

    Number
    get_quadratic_coefficient_derivative(
      const unsigned int            component_i,
      const unsigned int            component_j,
      const dealii::Vector<Number> &values) const
    {
      return Q_derivative[component_i][component_j](values);
    }

  private:
    // number of species in the diffusion problem
    unsigned int n_species{0};
    // flag indicating if values are constant or not
    bool constant{false};

    // linear term parameter
    ConstantCoefficientMatrix<Number> K;
    // higher order terms
    HigherOrderMatrix<Number> Q;
    // derivative of higher order terms
    HigherOrderMatrix<Number> Q_derivative;
  };
} // namespace Diffusion

#endif // MSHCMM_LOCAL_DIFFUSION_PARAMETERS_H
