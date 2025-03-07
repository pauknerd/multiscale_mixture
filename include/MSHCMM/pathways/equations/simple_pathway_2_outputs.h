#pragma once

#include "equation_helpers.h"
#include "pathway_equation_base.h"


namespace Pathways::Equations
{
  using namespace EquationHelpers;

  template <typename Number = double>
  class SimplePathway2Outputs : public PathwayEquationBase<Number>
  {
  public:
    SimplePathway2Outputs() = default;

    ODE<Number> &
    get_ODE() override
    {
      return ode;
    };

    [[nodiscard]] unsigned int
    n_components() const override
    {
      return n_comps;
    }

    [[nodiscard]] unsigned int
    n_inputs() const override
    {
      return n_input_nodes;
    }

    [[nodiscard]] unsigned int
    n_outputs() const override
    {
      return n_output_nodes;
    }

    [[nodiscard]] const std::vector<std::string> &
    get_node_names() const override
    {
      return node_names;
    }

  private:
    unsigned int n_comps        = 3;
    unsigned int n_input_nodes  = 1;
    unsigned int n_output_nodes = 2;

    std::vector<std::string> node_names{"stress_input", "col", "proliferation"};

    // parameters of the pathway equation
    const Number tau   = 1.0;
    const Number w     = 1.0;
    const Number EC_50 = 0.5;
    const Number n     = 1.4;
    const Number Ymax  = 1.0;

    // actual pathway equation
    ODE<Number> ode = [&](Number                        t,
                          const dealii::Vector<Number> &y,
                          dealii::Vector<Number>       &ydot) -> int {
      (void)t;

      ydot[0] = 0.0; //-0.0 * y[0];
      ydot[1] = 1 / tau * (act(y[0], w, n, EC_50) * Ymax - y[1]);
      ydot[2] = 1 / tau * (act(y[0], w, n, EC_50) * Ymax - y[2]);

      return 0;
    };
  };
} // namespace Pathways::Equations
