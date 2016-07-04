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
        armd.Calc_Forces<moleculardynamics::ParallelType::NoParallel>();
        armd.Move_Atoms<moleculardynamics::ParallelType::NoParallel>();
    }

    cp.checkpoint("���񉻖���", __LINE__);

    armd.reset();
    
    cp.checkpoint("�ď�����", __LINE__);

    for (auto i = 0; i < LOOP; i++) {
        armd.Calc_Forces<moleculardynamics::ParallelType::Cilk>();
        armd.Move_Atoms<moleculardynamics::ParallelType::Cilk>();
    }

    cp.checkpoint("TBB�ŕ���", __LINE__);

    armd.reset();

    cp.checkpoint("�ď�����", __LINE__);

    for (auto i = 0; i < LOOP; i++) {
        armd.Calc_Forces<moleculardynamics::ParallelType::OpenCl>();
        armd.Move_Atoms<moleculardynamics::ParallelType::OpenCl>();
    }

    cp.checkpoint("OpenCL�ŕ���", __LINE__);

    cp.checkpoint_print();
    
    armd.getinfo();

    return 0;
}