#include <deal.II/base/exceptions.h>

#include <MSHCMM/pathways/id_distributions.h>

#include <functional>
#include <random>

namespace Pathways::IdDistributions
{

  //! uniform distribution
  Uniform::Uniform(const unsigned int seed)
    : generator(seed)
  {}

  unsigned int
  Uniform::operator()(const unsigned int n_pathways)
  {
    // setup random number generator
    std::uniform_int_distribution<unsigned int> pathway_selector(0,
                                                                 n_pathways -
                                                                   1);
    // draw random number for pathway_id of cell
    return pathway_selector(generator);
  }


  Normal::Normal(const unsigned int seed,
                 const double       mean,
                 const double       std_dev)
    : generator(seed)
    , mean(mean)
    , std_dev(std_dev)
  {}

  unsigned int
  Normal::operator()(const unsigned int n_pathways)
  {
    // check if we got a user-specified mean and standard deviation
    const double mean_    = mean < 0.0 ? 0.5 * n_pathways : mean;
    const double std_dev_ = std_dev < 0.0 ? 0.5 * n_pathways / 3.0 : std_dev;

    Assert(mean_ > 0.0,
           dealii::ExcMessage(
             "Mean of pathway id distribution must be larger than 0.0!"));
    Assert(
      mean_ < n_pathways,
      dealii::ExcMessage(
        "Mean of pathway id distribution can not be larger than the number of pathways!"));
    Assert(
      std_dev_ > 0.0,
      dealii::ExcMessage(
        "Standard deviation of pathway id distribution must be larger than 0.0!"));

    // setup random number generator
    std::normal_distribution<> pathway_selector(mean_, std_dev_);
    // draw random number for pathway_id of cell
    // round to the nearest integer
    auto pathway_id =
      static_cast<unsigned int>(std::round(pathway_selector(generator)));
    // check that pathway_is is valid, i.e., is between 0 and n_pathways-1
    if (pathway_id < 0)
      pathway_id = 0;
    if (pathway_id >= n_pathways)
      pathway_id = n_pathways - 1;
    return pathway_id;
  }


  Binomial::Binomial(const unsigned int seed)
    : generator(seed)
  {}

  unsigned int
  Binomial::operator()(const unsigned int n_pathways)
  {
    // setup random number generator
    std::binomial_distribution<unsigned int> pathway_selector(
      n_pathways, (n_pathways - 1.0) / (2.0 * n_pathways));
    // draw random number for pathway_id of cell
    // return static_cast<unsigned int>(pathway_selector(generator));
    return (pathway_selector(generator));
  }


  ConstantId::ConstantId(const unsigned int id)
    : id(id)
  {}

  unsigned int
  ConstantId::operator()(const unsigned int n_pathways)
  {
    (void)n_pathways;
    return id;
  }
} // namespace Pathways::IdDistributions