#pragma once

#include <deal.II/base/function.h>

namespace Common
{
  /**
   * Helper Functions for common boundary conditions.
   */
  template <int dim, typename Number = double>
  class BoundaryConditionLinearRamp : public dealii::Function<dim, Number>
  {
  public:
    BoundaryConditionLinearRamp(const Number              t_start,
                                const Number              t_end,
                                const std::vector<Number> values)
      : dealii::Function<dim, Number>(values.size())
      , t_start(t_start)
      , t_end(t_end)
      , values(values)
    {}

    void
    vector_value(const dealii::Point<dim> &p,
                 dealii::Vector<Number>   &value) const override
    {
      (void)p;
      if (t_start <= this->get_time() and this->get_time() <= t_end)
        {
          for (unsigned int i = 0; i < values.size(); ++i)
            {
              value[i] = values[i] * (this->get_time() - t_start);
            }
        }
      else if (this->get_time() < t_start)
        {
          for (unsigned int i = 0; i < values.size(); ++i)
            {
              value[i] = 0.0;
            }
        }
      else if (this->get_time() > t_end)
        {
          for (unsigned int i = 0; i < values.size(); ++i)
            {
              value[i] = values[i] * (t_end - t_start);
            }
        }
    }

  private:
    Number t_start{0.0};
    Number t_end{0.0};

    std::vector<Number> values;
  };

  template <int dim, typename Number = double>
  class ExponentialPressureRamp : public dealii::Function<dim, Number>
  {
  public:
    ExponentialPressureRamp(const Number base_pressure,
                            const Number final_pressure,
                            const Number k)
      : dealii::Function<dim, Number>(dim)
      , base_pressure(base_pressure)
      , final_pressure(final_pressure)
      , k(k)
    {
      pressure_increment = final_pressure - base_pressure;
    }

    Number
    value(const dealii::Point<dim> &p,
          const unsigned int        component = 0) const override
    {
      (void)p;
      (void)component;

      return -(base_pressure +
               pressure_increment * (1.0 - std::exp(-k * this->get_time())));
    }

  private:
    Number base_pressure;
    Number final_pressure;
    Number pressure_increment;

    Number k;
  };

} // namespace Common
