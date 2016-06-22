#include <array>                                // for std::array
#include <boost/compute/types/fundamental.hpp>  // for boost::compute::float4_
#include <boost/range/algorithm/fill.hpp>       // for boost::fill

namespace utility {
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

	    boost::compute::float4_& operator=(boost::compute::float4_ & lhs) {
            data = { lhs[0], lhs[1], lhs[2], lhs[3] };

	        return lhs;
	    }
	};
}