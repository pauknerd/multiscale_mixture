#pragma once

#include <deal.II/lac/vector.h>

#include <functional>
#include <string>
#include <vector>

namespace Pathways
{
  namespace Equations
  {

    //! Pathway equation.
    template <typename Number>
    using ODE = std::function<int(double                        t,
                                  const dealii::Vector<Number> &y,
                                  dealii::Vector<Number>       &ydot)>;

    template <typename Number = double>
    class PathwayEquationBase
    {
    public:
      virtual ~PathwayEquationBase() = default;

      // this is the central function that needs to be implemented by derived
      // classes
      virtual ODE<Number> &
      get_ODE() = 0;

      // get the total number of components of the system of equations
      [[nodiscard]] virtual unsigned int
      n_components() const = 0;

      // get the number of inputs of the system of equations
      [[nodiscard]] virtual unsigned int
      n_inputs() const = 0;

      // get the number of outputs of the system of equations
      [[nodiscard]] virtual unsigned int
      n_outputs() const = 0;

      // get names of nodes
      [[nodiscard]] virtual const std::vector<std::string> &
      get_node_names() const = 0;
    };

  } // namespace Equations
} // namespace Pathways
