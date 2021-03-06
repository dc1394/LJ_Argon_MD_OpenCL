#include "checkpoint.h"
#include "moleculardynamics/Ar_moleculardynamics.h"

namespace {
    static auto constexpr LOOP = 100;
}

int main()
{
    checkpoint::CheckPoint cp;
    cp.checkpoint("処理開始", __LINE__);

    moleculardynamics::Ar_moleculardynamics<float> armd;

    cp.checkpoint("初期化処理", __LINE__);

    for (auto i = 0; i < LOOP; i++) {
        armd.Calc_Forces<moleculardynamics::ParallelType::NoParallel>();
        armd.Move_Atoms<moleculardynamics::ParallelType::NoParallel>();
    }

    cp.checkpoint("並列化無し", __LINE__);

    armd.reset();
    
    cp.checkpoint("再初期化", __LINE__);

    for (auto i = 0; i < LOOP; i++) {
        armd.Calc_Forces<moleculardynamics::ParallelType::Tbb>();
        armd.Move_Atoms<moleculardynamics::ParallelType::Tbb>();
    }

    cp.checkpoint("TBBで並列化", __LINE__);

    armd.reset();

    cp.checkpoint("再初期化", __LINE__);

    for (auto i = 0; i < LOOP; i++) {
        armd.Calc_Forces<moleculardynamics::ParallelType::OpenCl>();
        armd.Move_Atoms<moleculardynamics::ParallelType::OpenCl>();
    }

    cp.checkpoint("OpenCLで並列化", __LINE__);

    cp.checkpoint_print();
    
    armd.getinfo();

    return 0;
}