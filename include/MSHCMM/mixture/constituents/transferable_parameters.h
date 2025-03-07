#pragma once

namespace Mixture::Constituents
{
  /**
   * Parameters that can be transferred from the diffusion or pathway problem to
   * the HCMM and vice versa.
   *
   * @tparam Number
   */
  template <typename Number = double>
  struct TransferableParameters
  {
    // these parameters are set by other problems
    // true mass production rate; output of the pathway scaled by constituent
    // densities and maybe by time step
    Number mass_production{0.0};
    // current mass fraction of constituent (coming from diffusion problem)
    Number current_mass_fraction{0.0};

    Number decay_time{101.0}; // only used by classical HCMM

    // todo: how should this be initialized?
    Number contractility{0.5};
    // this parameter is transferred to other problems
    Number fiber_cauchy_stress{0.0};
    Number active_fiber_cauchy_stress{0.0};

    Number trace_sigma_h{0.0};

    // parameters only relevant for outputting?
    Number sigma_hom{0.0};
    Number lambda_rem{0.0};

    TransferableParameters<Number> &
    operator+=(const TransferableParameters<Number> &rhs)
    {
      // just add up all the individual components
      this->mass_production += rhs.mass_production;
      this->current_mass_fraction += rhs.current_mass_fraction;
      this->decay_time += rhs.decay_time;
      this->contractility += rhs.contractility;
      this->fiber_cauchy_stress += rhs.fiber_cauchy_stress;
      this->sigma_hom += rhs.sigma_hom;
      this->lambda_rem += rhs.lambda_rem;
      this->trace_sigma_h += rhs.trace_sigma_h;

      return *this;
    }

    TransferableParameters<Number> &
    operator/=(const unsigned int n)
    {
      // just add up all the individual components
      this->mass_production /= n;
      this->current_mass_fraction /= n;
      this->decay_time /= n;
      this->contractility /= n;
      this->fiber_cauchy_stress /= n;
      this->sigma_hom /= n;
      this->lambda_rem /= n;
      this->trace_sigma_h /= n;

      return *this;
    }
  };
} // namespace Mixture::Constituents
