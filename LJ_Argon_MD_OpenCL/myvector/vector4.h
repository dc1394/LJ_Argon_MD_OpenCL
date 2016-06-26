#include <array>                                // for std::array
#include <boost/compute/types/fundamental.hpp>  // for boost::compute::float4_
#include <boost/range/algorithm/fill.hpp>       // for boost::fill

namespace myvector {
	template <typename T>
	struct Vector4 final {
	public:
		std::array<T, 4> data;

		Vector4()
		{
            boost::fill(data, static_cast<T>(0));
		}

		Vector4(T x, T y, T z)
		{
            data = { x, y, z, static_cast<T>(0) };
		}

	    boost::compute::float4_& operator=(boost::compute::float4_ & rhs) {
            data = { rhs[0], rhs[1], rhs[2], rhs[3] };
	        return rhs;
	    }
	};
}