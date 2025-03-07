#pragma once

#include <functional>
#include <random>

namespace Pathways::IdDistributions
{
  //! Type alias
  // using PathwayIdDistribution = std::function<unsigned int(const unsigned int
  // n_pathways)>;

  //! uniform distribution
  class Uniform
  {
  public:
    explicit Uniform(const unsigned int seed = 1);

    unsigned int
    operator()(const unsigned int n_pathways);

  private:
    std::default_random_engine generator;
  };


  // PathwayIdDistribution uniform(const unsigned int seed = 1);

  /**
   *  Normal distribution of pathways. If no mean or standard deviation is
   * passed a normal distribution with mean 0.5 * n_pathways and std. dev. 1/3 *
   * mean is assumed. The implementation is based on a continuous normal
   * distribution and std::round() is used to get the nearest integer value. The
   * passed mean and standard deviation are not checked for feasibility! Only
   * basic checks are performed: 0.0 < mean < n_pathways, std_dev > 0.0.
   */
  // PathwayIdDistribution normal(const unsigned int seed = 1, const double mean
  // = -1.0, const double std_dev = -1.0);
  class Normal
  {
  public:
    Normal(const unsigned int seed    = 1,
           const double       mean    = -1.0,
           const double       std_dev = -1.0);

    unsigned int
    operator()(const unsigned int n_pathways);

  private:
    std::default_random_engine generator;

    const double mean;
    const double std_dev;
  };

  //! binomial distribution
  // PathwayIdDistribution binomial(const unsigned int seed = 1);
  class Binomial
  {
  public:
    explicit Binomial(const unsigned int seed = 1);

    unsigned int
    operator()(const unsigned int n_pathways);

  private:
    std::default_random_engine generator;
  };

  /**
   * Just returns a constant id which can be specified in the constructor. Note
   * that if the id is set to -1, at every time step a random pathway is picked
   * and solved from all the available pathways.
   */
  class ConstantId
  {
  public:
    explicit ConstantId(const unsigned int id);

    unsigned int
    operator()(const unsigned int n_pathways);

  private:
    const unsigned int id;
  };
  // PathwayIdDistribution constant_id(const unsigned int id, const unsigned int
  // seed = 1);
} // namespace Pathways::IdDistributions
