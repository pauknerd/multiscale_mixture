#pragma once

#include <deal.II/base/exceptions.h>

#include <string>


namespace Pathways::Cells
{

  enum class CellType
  {
    FBS,
    SMC,
    MP,
    EC,
    none
  };

  inline std::string
  CellType2string(const CellType &cell_type)
  {
    switch (cell_type)
      {
        case CellType::FBS:
          return "FBS";
        case CellType::SMC:
          return "SMC";
        case CellType::MP:
          return "MP";
        case CellType::EC:
          return "EC";
        case CellType::none:
          return "none";
        default:
          AssertThrow(
            false,
            dealii::ExcMessage(
              "CellType cannot be converted to string! Maybe the conversion is missing..."));
      }
  }
} // namespace Pathways::Cells
