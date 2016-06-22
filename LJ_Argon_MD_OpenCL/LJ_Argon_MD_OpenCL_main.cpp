#include "moleculardynamics/Ar_moleculardynamics.h"

int main()
{
    moleculardynamics::Ar_moleculardynamics<float> armd;
    
    for (auto i = 0; i < 1000; i++) {
        armd.Calc_Forces();
        armd.Move_Atoms();
    }

    return 0;
}