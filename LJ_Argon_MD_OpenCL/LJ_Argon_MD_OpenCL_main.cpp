#include "moleculardynamics/Ar_moleculardynamics.h"
#include <chrono>
#include <iostream>

int main()
{
    moleculardynamics::Ar_moleculardynamics<float> armd;
    
    std::chrono::time_point<std::chrono::system_clock> begin(std::chrono::system_clock::now());
    for (auto i = 0; i < 20; i++) {
        armd.Calc_Forces();
        armd.Move_Atoms();
    }

    const auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

    return 0;
}