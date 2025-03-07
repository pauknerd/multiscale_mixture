#pragma once

#include "equation_helpers.h"
#include "pathway_equation_base.h"

namespace Pathways
{
  namespace Equations
  {
    using namespace EquationHelpers;

    template <typename Number = double>
    class Irons2020ECPathway : public PathwayEquationBase<Number>
    {
    public:
      Irons2020ECPathway()
      {
        for (unsigned int i = 0; i < n_comps; i++)
          {
            node_names.push_back(dealii::Utilities::to_string(i));
          }
      }

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
      unsigned int n_comps        = 4;
      unsigned int n_input_nodes  = 1;
      unsigned int n_output_nodes = 2;

      // species ids
      int Wss  = 0;
      int ETBR = 1;
      int NO   = 2;
      int ET1  = 3;

      // create vector with pathway nodes
      std::vector<std::string> node_names;

      // parameters of the pathway equation
      const Number tau  = 1.0;
      const Number w    = 1.0;
      const Number EC50 = 0.55;
      const Number n    = 1.25;
      const Number ymax = 1.0;

      // actual pathway equation
      ODE<Number> ode = [&](Number                        t,
                            const dealii::Vector<Number> &y,
                            dealii::Vector<Number>       &ydot) -> int {
        (void)t;

        // input
        ydot[Wss] = 0.0;
        // intracellular
        ydot[ETBR] = (act(y[ET1], w, n, EC50) * ymax - y[ETBR]) / tau;
        // outputs
        ydot[NO] =
          (OR(act(y[Wss], w, n, EC50), act(y[ETBR], w, n, EC50)) * ymax -
           y[NO]) /
          tau;
        ydot[ET1] =
          (AND(inhib(y[Wss], w, n, EC50), inhib(y[NO], w, n, EC50)) * ymax -
           y[ET1]) /
          tau;

        return 0;
      };
    };
  } // namespace Equations
} // namespace Pathways
