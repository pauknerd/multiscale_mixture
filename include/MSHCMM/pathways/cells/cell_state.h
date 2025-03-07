#pragma once

#include <deal.II/lac/vector.h>


// todo: maybe add specific functions to get input and output from cell state?
//  based on iterators? E.g., n_inputs, n_outputs and then return an iterator
//  range cell_state.begin(), cell_state.begin() + n_inputs?


namespace Pathways::Cells
{

  template <typename Number = double>
  struct CellState
  {
  public:
    /**
     * Constructor for cell state vector, assumes first n_inputs entries are
     * inputs and last n_outputs entries are outputs
     */
    explicit CellState(const unsigned int n_components,
                       const unsigned int n_inputs,
                       const unsigned int n_outputs)
      : cell_state(n_components)
      , n_inputs(n_inputs)
      , n_outputs(n_outputs)
    {
      initialized = true;
    }

    explicit CellState(const dealii::Vector<Number> &initial_state,
                       const unsigned int            n_inputs,
                       const unsigned int            n_outputs)
      : cell_state(initial_state)
      , n_inputs(n_inputs)
      , n_outputs(n_outputs)
    {
      initialized = true;
    }

    bool                   initialized = false;
    dealii::Vector<Number> cell_state;

    unsigned int n_inputs;
    unsigned int n_outputs;
  };
} // namespace Pathways::Cells
