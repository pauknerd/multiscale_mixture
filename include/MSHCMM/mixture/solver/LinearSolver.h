#pragma once

#include "utility"

template <typename VectorType, typename MatrixType>
class DirectSolver
{
public:
  explicit DirectSolver(const dealii::SolverControl &solver_control =
                          dealii::SolverControl(1000, 1.e-8))
    : solver_control(solver_control)
  {}

  std::pair<unsigned int, double>
  solve(const MatrixType &A, VectorType &x, const VectorType &b)
  {
    return solver_internal(A, x, b);
  }

  std::pair<unsigned int, double>
  solve(const typename MatrixType::BlockType &A,
        typename VectorType::BlockType       &x,
        const typename VectorType::BlockType &b)
  {
    return solver_internal(A, x, b);
  }

  dealii::SolverControl &
  get_solver_control()
  {
    return solver_control;
  }

private:
  template <typename M, typename V>
  std::pair<unsigned int, double>
  solver_internal(const M &A, V &x, const V &b)
  {
    typename dealii::TrilinosWrappers::SolverDirect::AdditionalData
      additional_data;

    dealii::TrilinosWrappers::SolverDirect solver(solver_control,
                                                  additional_data);
    solver.solve(A, x, b);

    return std::pair<unsigned int, double>{solver_control.last_step(),
                                           solver_control.last_value()};
  }

  dealii::SolverControl solver_control;
};

template <typename VectorType, typename MatrixType>
class IterativeSolver
{
public:
  explicit IterativeSolver(const std::string           &solver_type,
                           const dealii::SolverControl &solver_control =
                             dealii::SolverControl(1000, 1.e-8))
    : solver_type(solver_type)
    , solver_control(solver_control)
  {}

  template <typename Preconditioner>
  std::pair<unsigned int, double>
  solve(const MatrixType     &A,
        VectorType           &x,
        const VectorType     &b,
        const Preconditioner &preconditioner)
  {
    return solver_internal(A, x, b, preconditioner);
  }

  template <typename Preconditioner>
  std::pair<unsigned int, double>
  solve(const typename MatrixType::BlockType &A,
        typename VectorType::BlockType       &x,
        const typename VectorType::BlockType &b,
        const Preconditioner                 &preconditioner)
  {
    return solver_internal(A, x, b, preconditioner);
  }

  dealii::SolverControl &
  get_solver_control()
  {
    return solver_control;
  }

private:
  template <typename M, typename V, typename P>
  std::pair<unsigned int, double>
  solver_internal(const M &A, V &x, const V &b, const P &p)
  {
    //    const auto   solver_its = static_cast<unsigned int>(A.m());
    //    const double tol_sol    = 1.e-6 * b.l2_norm();
    //
    //    solver_control          = dealii::SolverControl(solver_its, tol_sol);

    dealii::SolverSelector<V> solver(solver_type, solver_control);
    solver.solve(A, x, b, p);

    return std::pair<unsigned int, double>{solver_control.last_step(),
                                           solver_control.last_value()};
  }

  std::string           solver_type{"unknown"};
  dealii::SolverControl solver_control;
};
