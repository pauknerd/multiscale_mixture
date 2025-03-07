#pragma once

#include "equation_helpers.h"
#include "pathway_equation_base.h"


namespace Pathways::Equations
{
  using namespace EquationHelpers;

  template <typename Number = double>
  class PathwayIrons2020 : public PathwayEquationBase<Number>
  {
  public:
    PathwayIrons2020()
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
    unsigned int n_comps        = 50;
    unsigned int n_input_nodes  = 5;
    unsigned int n_output_nodes = 4;

    // species ids
    int Stress             = 0;
    int Wss                = 1;
    int AngIIin            = 2;
    int SACs               = 3;
    int Integrins          = 4;
    int PDGF               = 5;
    int AngII              = 6;
    int latentTGFb1        = 7;
    int TGFb1              = 8;
    int TGFbR2             = 9;
    int TGFbR1             = 10;
    int pSmad23            = 11;
    int Smad4              = 12;
    int Smad7              = 13;
    int TSP1               = 14;
    int TIMP               = 15;
    int p38                = 16;
    int JNK                = 17;
    int ERK                = 18;
    int MMP1               = 19;
    int MMP2               = 20;
    int MMP9               = 21;
    int AT1R               = 22;
    int AT2R               = 23;
    int PDGFR              = 24;
    int NO                 = 25;
    int ET1                = 26;
    int ETAR               = 27;
    int ETBR               = 28;
    int PI3K               = 29;
    int Akt                = 30;
    int mTOR               = 31;
    int mTORC1             = 32;
    int mTORC2             = 33;
    int p70S6K             = 34;
    int Ca                 = 35;
    int MLCK               = 36;
    int Myosin             = 37;
    int FAK                = 38;
    int Cdc42              = 39;
    int Arp23              = 40;
    int RhoA               = 41;
    int ROCK               = 42;
    int Actin              = 43;
    int Col1mRNA           = 44;
    int Col3mRNA           = 45;
    int Col1               = 46;
    int Col3               = 47;
    int ActomyosinActivity = 48;
    int SMCproliferation   = 49;

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

      // inputs
      ydot[Stress]  = 0.; //(w[0]*ymax[Stress] - y[Stress])/tau[Stress] ;
      ydot[Wss]     = 0.; //(w[1]*ymax[Wss] - y[Wss])/tau[Wss] ;
      ydot[AngIIin] = 0.; //(w[2]*ymax[AngIIin] - y[AngIIin])/tau[AngIIin] ;
      ydot[SACs]    = 0.; //(w[3]*ymax[SACs] - y[SACs])/tau[SACs] ;
      ydot[Integrins] =
        0.; //(w[4]*ymax[Integrins] - y[Integrins])/tau[Integrins] ;
      // intermediate values
      ydot[PDGF] = (act(y[Stress], w, n, EC50) * ymax - y[PDGF]) / tau;
      ydot[AngII] =
        (OR(act(y[AngIIin], w, n, EC50), act(y[Stress], w, n, EC50)) * ymax -
         y[AngII]) /
        tau;
      ydot[latentTGFb1] =
        (act(y[Stress], w, n, EC50) * ymax - y[latentTGFb1]) / tau;
      ydot[TGFb1] =
        (OR(AND(act(y[Stress], w, n, EC50),
                act(y[Integrins], w, n, EC50),
                act(y[latentTGFb1], w, n, EC50)),
            OR(AND(act(y[latentTGFb1], w, n, EC50), act(y[MMP2], w, n, EC50)),
               OR(AND(act(y[latentTGFb1], w, n, EC50),
                      act(y[MMP9], w, n, EC50)),
                  AND(act(y[latentTGFb1], w, n, EC50),
                      act(y[TSP1], w, n, EC50))))) *
           ymax -
         y[TGFb1]) /
        tau;
      ydot[TGFbR2] = (act(y[TGFb1], w, n, EC50) * ymax - y[TGFbR2]) / tau;
      ydot[TGFbR1] =
        (AND(act(y[TGFbR2], w, n, EC50), inhib(y[Smad7], w, n, EC50)) * ymax -
         y[TGFbR1]) /
        tau;
      ydot[pSmad23] =
        (OR(act(y[TGFbR1], w, n, EC50), act(y[AT1R], w, n, EC50)) * ymax -
         y[pSmad23]) /
        tau;
      ydot[Smad4] = (act(y[pSmad23], w, n, EC50) * ymax - y[Smad4]) / tau;
      ydot[Smad7] = (act(y[Smad4], w, n, EC50) * ymax - y[Smad7]) / tau;
      ydot[TSP1]  = (OR(act(y[p38], w, n, EC50),
                       OR(act(y[ERK], w, n, EC50), act(y[JNK], w, n, EC50))) *
                      ymax -
                    y[TSP1]) /
                   tau;
      ydot[TIMP] = (act(y[Smad4], w, n, EC50) * ymax - y[TIMP]) / tau;
      ydot[p38] =
        (OR(act(y[TGFbR1], w, n, EC50),
            OR(act(y[PDGFR], w, n, EC50),
               OR(act(y[AT1R], w, n, EC50), act(y[FAK], w, n, EC50)))) *
           ymax -
         y[p38]) /
        tau;
      ydot[JNK] =
        (OR(act(y[TGFbR1], w, n, EC50),
            OR(act(y[PDGFR], w, n, EC50),
               OR(act(y[AT1R], w, n, EC50), act(y[FAK], w, n, EC50)))) *
           ymax -
         y[JNK]) /
        tau;
      ydot[ERK] =
        (OR(AND(act(y[TGFbR1], w, n, EC50), inhib(y[AT2R], w, n, EC50)),
            OR(AND(inhib(y[AT2R], w, n, EC50), act(y[PDGFR], w, n, EC50)),
               OR(AND(inhib(y[AT2R], w, n, EC50), act(y[ETAR], w, n, EC50)),
                  OR(AND(act(y[AT1R], w, n, EC50), inhib(y[AT2R], w, n, EC50)),
                     AND(inhib(y[AT2R], w, n, EC50),
                         act(y[FAK], w, n, EC50)))))) *
           ymax -
         y[ERK]) /
        tau;
      ydot[MMP1] =
        (AND(inhib(y[TIMP], w, n, EC50), act(y[p38], w, n, EC50)) * ymax -
         y[MMP1]) /
        tau;
      ydot[MMP2] =
        (OR(AND(inhib(y[TIMP], w, n, EC50), act(y[p38], w, n, EC50)),
            OR(AND(inhib(y[TIMP], w, n, EC50), act(y[ERK], w, n, EC50)),
               OR(AND(inhib(y[TIMP], w, n, EC50), act(y[JNK], w, n, EC50)),
                  AND(inhib(y[TIMP], w, n, EC50), act(y[Akt], w, n, EC50))))) *
           ymax -
         y[MMP2]) /
        tau;
      ydot[MMP9] =
        (OR(AND(inhib(y[TIMP], w, n, EC50), act(y[p38], w, n, EC50)),
            AND(inhib(y[TIMP], w, n, EC50), act(y[ERK], w, n, EC50))) *
           ymax -
         y[MMP9]) /
        tau;
      ydot[AT1R]  = (act(y[AngII], w, n, EC50) * ymax - y[AT1R]) / tau;
      ydot[AT2R]  = (act(y[AngII], w, n, EC50) * ymax - y[AT2R]) / tau;
      ydot[PDGFR] = (act(y[PDGF], w, n, EC50) * ymax - y[PDGFR]) / tau;
      ydot[NO]    = (OR(act(y[Wss], w, n, EC50),
                     OR(act(y[ETBR], w, n, EC50), act(y[AT2R], w, n, EC50))) *
                    ymax -
                  y[NO]) /
                 tau;
      ydot[ET1] =
        (AND(act(y[Wss], w, n, EC50), inhib(y[NO], w, n, EC50)) * ymax -
         y[ET1]) /
        tau;
      ydot[ETAR] = (act(y[ET1], w, n, EC50) * ymax - y[ETAR]) / tau;
      ydot[ETBR] = (act(y[ET1], w, n, EC50) * ymax - y[ETBR]) / tau;
      ydot[PI3K] =
        (OR(act(y[PDGFR], w, n, EC50),
            OR(act(y[ETAR], w, n, EC50),
               OR(act(y[AT1R], w, n, EC50), act(y[TGFbR1], w, n, EC50)))) *
           ymax -
         y[PI3K]) /
        tau;
      ydot[Akt] =
        (OR(act(y[PI3K], w, n, EC50), act(y[mTORC2], w, n, EC50)) * ymax -
         y[Akt]) /
        tau;
      ydot[mTOR]   = (act(y[Akt], w, n, EC50) * ymax - y[mTOR]) / tau;
      ydot[mTORC1] = (act(y[mTOR], w, n, EC50) * ymax - y[mTORC1]) / tau;
      ydot[mTORC2] = (act(y[mTOR], w, n, EC50) * ymax - y[mTORC2]) / tau;
      ydot[p70S6K] = (act(y[mTORC1], w, n, EC50) * ymax - y[p70S6K]) / tau;
      ydot[Ca] =
        (OR(act(y[ETAR], w, n, EC50),
            OR(act(y[AT1R], w, n, EC50),
               AND(act(y[Stress], w, n, EC50), act(y[SACs], w, n, EC50)))) *
           ymax -
         y[Ca]) /
        tau;
      ydot[MLCK] =
        (OR(AND(inhib(y[NO], w, n, EC50), act(y[Ca], w, n, EC50)),
            AND(inhib(y[NO], w, n, EC50), act(y[ROCK], w, n, EC50))) *
           ymax -
         y[MLCK]) /
        tau;
      ydot[Myosin] = (act(y[MLCK], w, n, EC50) * ymax - y[Myosin]) / tau;
      ydot[FAK] =
        (AND(act(y[Stress], w, n, EC50), act(y[Integrins], w, n, EC50)) * ymax -
         y[FAK]) /
        tau;
      ydot[Cdc42] = (act(y[FAK], w, n, EC50) * ymax - y[Cdc42]) / tau;
      ydot[Arp23] = (act(y[Cdc42], w, n, EC50) * ymax - y[Arp23]) / tau;
      ydot[RhoA] =
        (OR(AND(inhib(y[AT2R], w, n, EC50), act(y[mTORC2], w, n, EC50)),
            OR(AND(act(y[AT1R], w, n, EC50), inhib(y[AT2R], w, n, EC50)),
               AND(act(y[Stress], w, n, EC50),
                   act(y[Integrins], w, n, EC50),
                   inhib(y[AT2R], w, n, EC50)))) *
           ymax -
         y[RhoA]) /
        tau;
      ydot[ROCK] = (act(y[RhoA], w, n, EC50) * ymax - y[ROCK]) / tau;
      ydot[Actin] =
        (OR(act(y[ROCK], w, n, EC50), act(y[Arp23], w, n, EC50)) * ymax -
         y[Actin]) /
        tau;
      ydot[Col1mRNA] =
        (OR(act(y[Smad4], w, n, EC50),
            OR(act(y[p38], w, n, EC50), act(y[ERK], w, n, EC50))) *
           ymax -
         y[Col1mRNA]) /
        tau;
      ydot[Col3mRNA] =
        (OR(act(y[Smad4], w, n, EC50), act(y[p38], w, n, EC50)) * ymax -
         y[Col3mRNA]) /
        tau;
      // outputs
      ydot[Col1] = (AND(inhib(y[MMP1], w, n, EC50),
                        inhib(y[MMP2], w, n, EC50),
                        act(y[Col1mRNA], w, n, EC50)) *
                      ymax -
                    y[Col1]) /
                   tau;
      ydot[Col3] = (AND(inhib(y[MMP1], w, n, EC50),
                        inhib(y[MMP2], w, n, EC50),
                        act(y[Col3mRNA], w, n, EC50)) *
                      ymax -
                    y[Col3]) /
                   tau;
      ydot[ActomyosinActivity] =
        (AND(act(y[Myosin], w, n, EC50), act(y[Actin], w, n, EC50)) * ymax -
         y[ActomyosinActivity]) /
        tau;
      ydot[SMCproliferation] =
        (OR(act(y[p70S6K], w, n, EC50), act(y[ERK], w, n, EC50)) * ymax -
         y[SMCproliferation]) /
        tau;

      return 0;
    };
  };
} // namespace Pathways::Equations
