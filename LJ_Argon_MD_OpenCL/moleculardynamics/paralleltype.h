/*! \file paralleltype.h
    \brief 並列化の手法を表す列挙型の宣言

    Copyright ©  2016 @dc1394 All Rights Reserved.
    This software is released under the BSD 2-Clause License.
*/

#ifndef _PARALLELTYPE_H_
#define _PARALLELTYPE_H_

#pragma once

#include <cstdint>  // for std::int32_t

namespace moleculardynamics {
    enum class ParallelType : std::int32_t {
        NoParallel = 0,
        OpenCl = 1,
        Tbb = 2
    };
}

#endif  // _PARALLELTYPE_H_
