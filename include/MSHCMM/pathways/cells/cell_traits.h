#pragma once

#include "cell_types.h"


namespace Pathways::Cells
{
  /**
   * Struct to store some information about the cell.
   */
  struct CellTraits
  {
  public:
    CellTraits(const CellType    &cell_type,
               const unsigned int pathway_id,
               const double       pathway_weight)
      : cell_type(cell_type)
      , pathway_id(pathway_id)
      , pathway_weight(pathway_weight)
    {}

    CellType     cell_type;
    unsigned int pathway_id;
    double       pathway_weight;
  };
} // namespace Pathways::Cells
