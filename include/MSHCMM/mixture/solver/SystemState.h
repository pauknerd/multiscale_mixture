#pragma once


template <typename VectorType, typename MatrixType>
struct SystemState
{
  MatrixType system_matrix;
  VectorType system_rhs;

  VectorType solution;
  VectorType old_solution;

  VectorType solution_delta;
  VectorType predictor;

  void
  reset_system()
  {
    system_matrix = 0.0;
    system_rhs    = 0.0;
  }

  VectorType
  get_total_solution() const
  {
    VectorType total_solution(solution);
    total_solution += solution_delta;
    // todo: not sure if needed
    total_solution.compress(dealii::VectorOperation::add);

    return total_solution;
  }

  void
  add_new_increment()
  {
    solution += solution_delta;
  }
};
