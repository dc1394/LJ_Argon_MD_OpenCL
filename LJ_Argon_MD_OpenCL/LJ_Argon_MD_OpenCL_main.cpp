#include "checkpoint.h"
#include "moleculardynamics/Ar_moleculardynamics.h"

namespace {
    static auto constexpr LOOP = 100;
}

int main()
{
    checkpoint::CheckPoint cp;
    cp.checkpoint("ˆ—ŠJn", __LINE__);

    moleculardynamics::Ar_moleculardynamics<float> armd;
    
    cp.checkpoint("‰Šú‰»ˆ—", __LINE__);

    for (auto i = 0; i < LOOP; i++) {
        armd.Calc_Forces();
        armd.Move_Atoms();
    }

    cp.checkpoint("TBB‚Å•À—ñ‰»", __LINE__);

    armd.reset();

    cp.checkpoint("Ä‰Šú‰»", __LINE__);

    for (auto i = 0; i < LOOP; i++) {
        armd.Calc_Forces_OpenCL();
        armd.Move_Atoms_OpenCL();
    }

    cp.checkpoint("OpenCL‚Å•À—ñ‰»", __LINE__);

    cp.checkpoint_print();
    
    armd.getinfo();

    return 0;
}