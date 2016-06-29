/*! \file Ar_moleculardynamics.h
    \brief 自作の4次元ベクトルクラスの宣言と実装

    Copyright ©  2016 @dc1394 All Rights Reserved.
    This software is released under the BSD 2-Clause License.
*/

#ifndef _VECTOR4_H_
#define _VECTOR4_H_

#include <array>                                // for std::array
#include <boost/compute/types/fundamental.hpp>  // for boost::compute::float4_
#include <boost/range/algorithm/fill.hpp>       // for boost::fill

namespace myvector {
    //! A template class.
    /*!
        自作の4次元ベクトルクラス
    */
	template <typename T>
	struct vector4 final {
	public:
		std::array<T, 4> data;

        //! A constructor.
        /*!
            デフォルトコンストラクタ
        */
		vector4()
		{
            boost::fill(data, static_cast<T>(0));
		}

        //! A constructor.
        /*!
            コンストラクタ
            \param x x成分
            \param y y成分
            \param z z成分
        */
		vector4(T x, T y, T z)
		{
            data = { x, y, z, static_cast<T>(0) };
		}

        //! A operator=.
        /*!
            演算子=
            \param rhs 右辺値
            \return rhsの参照をそのまま返す
        */
	    boost::compute::float4_& operator=(boost::compute::float4_ & rhs) {
            data = { rhs[0], rhs[1], rhs[2], rhs[3] };
	        return rhs;
	    }
	};
}

#endif  // _VECTOR4_H_
