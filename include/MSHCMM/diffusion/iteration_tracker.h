#pragma once

#include "arkode/arkode_arkstep.h"

struct IterationTracker
{
  void
  get_general_stats(void *arkode_mem)
  {
    ARKStepGetNumSteps(arkode_mem, &nsteps);
  }

  void
  get_nonlinsolvestats(void *arkode_mem)
  {
    ARKStepGetNonlinSolvStats(arkode_mem, &nniters, &nncfails);
  }

  void
  get_time_stepper_stats(void *arkode_mem)
  {
    ARKStepGetTimestepperStats(arkode_mem,
                               &expsteps,
                               &accsteps,
                               &step_attempts,
                               &nfe_evals,
                               &nfi_evals,
                               &nlinsetups,
                               &netfails);
  }

  void
  update_stats(void *arkode_mem)
  {
    get_general_stats(arkode_mem);
    get_nonlinsolvestats(arkode_mem);
    get_time_stepper_stats(arkode_mem);
  }

  void
  print_stats(dealii::ConditionalOStream &pcout)
  {
    // compute some performance metrics
    nl_iter_performance   = static_cast<double>(nniters) / nsteps;
    jac_quality           = static_cast<double>(lin_iters_jac) / nniters;
    imex_split_quality    = static_cast<double>(expsteps) / accsteps;
    time_stepping_quality = static_cast<double>(nsteps) / step_attempts;

    static const unsigned int l_width = 100;

    for (unsigned int i = 0; i < l_width; ++i)
      pcout << '_';
    pcout << std::endl;

    pcout << "Solved step in " << nniters - nniters_old
          << " non-linear iterations with " << nncfails - nncfails_old
          << " convergence fails." << std::endl;

    pcout << step_attempts - step_attempts_old << " steps attempted with "
          << nfe_evals - nfe_evals_old << " explicit function evaluations and "
          << nfi_evals - nfi_evals_old << " implicit function evaluations."
          << std::endl;

    //    pcout << "Non-linear solver performance: " << nl_iter_performance <<
    //    std::endl; pcout << "Jacobian quality: " << jac_quality << std::endl;
    //    pcout << "IMEX split quality: " << imex_split_quality << std::endl;
    //    pcout << "Time step adaptivity quality: " << time_stepping_quality <<
    //    std::endl;

    for (unsigned int i = 0; i < l_width; ++i)
      pcout << '_';
    pcout << std::endl;
  }

  void
  reset_stats()
  {
    nniters_old       = nniters;
    nncfails_old      = nncfails;
    expsteps_old      = expsteps;
    accsteps_old      = accsteps;
    step_attempts_old = step_attempts;
    nfe_evals_old     = nfe_evals;
    nfi_evals_old     = nfi_evals;
    nlinsetups_old    = nlinsetups;
    netfails_old      = netfails;
    nsteps_old        = nsteps;

    // lin_iters_jac = 0;
  }

  // number of nonlinear iterations performed
  long int nniters     = 0;
  long int nniters_old = 0;
  // number of nonlinear convergence failures
  long int nncfails     = 0;
  long int nncfails_old = 0;
  // expsteps – number of stability-limited steps taken in the solver.
  long int expsteps     = 0;
  long int expsteps_old = 0;
  // accsteps – number of accuracy-limited steps taken in the solver.
  long int accsteps     = 0;
  long int accsteps_old = 0;
  // step_attempts – number of steps attempted by the solver.
  long int step_attempts     = 0;
  long int step_attempts_old = 0;
  // nfe_evals – number of calls to the user-supplied explicit function
  long int nfe_evals     = 0;
  long int nfe_evals_old = 0;
  // nfi_evals – number of calls to the user-supplied implicit function.
  long int nfi_evals     = 0;
  long int nfi_evals_old = 0;
  // nlinsetups – number of linear solver setup calls made.
  long int nlinsetups     = 0;
  long int nlinsetups_old = 0;
  // netfails – number of error test failures.
  long int netfails     = 0;
  long int netfails_old = 0;
  // nnsteps
  long int nsteps     = 0;
  long int nsteps_old = 0;

  // linear solver iterations jacobian
  unsigned int lin_iters_jac = 0;

  //// ratios
  double nl_iter_performance   = 0;
  double jac_quality           = 0;
  double imex_split_quality    = 0;
  double time_stepping_quality = 0;
};
