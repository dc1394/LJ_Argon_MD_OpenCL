#include "checkpoint.h"
#include "moleculardynamics/Ar_moleculardynamics.h"

namespace {
    static auto constexpr LOOP = 100;
}

int main()
{
    checkpoint::CheckPoint cp;
    cp.checkpoint("�����J�n", __LINE__);

    moleculardynamics::Ar_moleculardynamics<float> armd;
    
    cp.checkpoint("����������", __LINE__);

    for (auto i = 0; i < LOOP; i++) {
        armd.Calc_Forces();
        armd.Move_Atoms();
    }

    cp.checkpoint("TBB�ŕ���", __LINE__);

    armd.reset();

    cp.checkpoint("�ď�����", __LINE__);

    for (auto i = 0; i < LOOP; i++) {
        armd.Calc_Forces_OpenCL();
        armd.Move_Atoms_OpenCL();
    }

    cp.checkpoint("OpenCL�ŕ���", __LINE__);

    cp.checkpoint_print();
    
    armd.getinfo();

    return 0;
}