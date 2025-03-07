#pragma once

struct Errors
{
  Errors()
    : norm(1.0)
    , u(1.0)
  {}

  void
  reset()
  {
    norm = 1.0;
    u    = 1.0;
  }
  void
  normalize(const Errors &rhs)
  {
    if (rhs.norm != 0.0)
      norm /= rhs.norm;
    if (rhs.u != 0.0)
      u /= rhs.u;
  }

  double norm, u;
};
