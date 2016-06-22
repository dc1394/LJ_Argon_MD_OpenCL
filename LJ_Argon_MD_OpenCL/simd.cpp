#define _SCL_SECURE_NO_WARNINGS

#include "vector4.h"
#include <iostream>
#include <memory>
#include <chrono>
#include <vector>

#include <boost/compute/container/vector.hpp>
#include <boost/compute/utility/source.hpp>
#include <boost/range/algorithm/generate.hpp>
#include <boost/range/algorithm/copy.hpp>


#ifdef _MCS_VER
#pragma warning(pop)
#endif

class Timer
{
	typedef std::chrono::time_point<std::chrono::system_clock> time_point;

	time_point begin;

public:
	void Start()
	{
		this->begin = std::chrono::system_clock::now();
	}

	std::chrono::milliseconds Time()
	{
		const auto end = std::chrono::system_clock::now();
		return std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	}
};

// なにもしない
template <typename T>
static void Normal(std::vector<utility::Vector4<T>> & x, std::vector<utility::Vector4<T>> & v, std::vector<utility::Vector4<T>> & f, T const m, T const dt, std::size_t const n)
{
	const auto tmp = dt*dt / 2.0f;
	const auto rm = 1.0f / m;

	for (auto i = 0; i < n; i++)
	{
		for (auto j = 0; j < 4; j++)
		{
			f[i].data[j] = 0;
		}

		for (auto j = 0U; j < n; j++)
		{
			if (i != j)
			{
				utility::Vector4<T> r;
				for (auto k = 0; k < 4; k++)
				{
					r.data[k] = x[j].data[k] - x[i].data[k];
				}

				T r2 = 0.0;
				for (int k = 0; k < 4; k++)
				{
					r2 += r.data[k] * r.data[k];
				}
				auto const r3 = r2 * std::sqrt(r2);

				for (auto k = 0; k < 4; k++)
				{
					f[i].data[k] += r.data[k] / r3;
				}
			}
		}
	}

	for (auto i = 0U; i < n; i++)
	{
		// a = f/m
		utility::Vector4<T> a;
		for (auto j = 0; j < 4; j++)
		{
			a.data[j] = f[i].data[j] * rm;
		}

		// x += v*dt + a*dt*dt/2
		for (auto j = 0; j < 4; j++)
		{
			auto const dxv = v[i].data[j] * dt;
			auto const dxa = a.data[j] * tmp;
			auto const dx = dxv + dxa;
			x[i].data[j] += dx;
		}

		// v += a*dt
		for (auto j = 0; j < 4; j++)
		{
			auto const dv = a.data[j] * dt;
			v[i].data[j] += dv;
		}
	}
}

#ifdef _OPENMP
// なにもしない+OpenMP
static void NormalOmp(Vector4 x[], Vector4 v[], Vector4 f[], const float m, const float dt, const std::size_t n)
{
	const float tmp = dt*dt / 2;
	const float rm = 1.0 / m;

	#pragma omp parallel
	{
		#pragma omp for
		for (int i = 0; i < n; i++)
		{
			// f = Σ(x_j - x)/|x_j - x|^3
			for (int j = 0; j < 4; j++)
			{
				f[i].data[j] = 0;
			}

			for (int j = 0; j < n; j++)
			{
				if (i != j)
				{
					Vector4 r;
					for (int k = 0; k < 4; k++)
					{
						r.data[k] = x[j].data[k] - x[i].data[k];
					}

					float r2 = 0;
					for (int k = 0; k < 4; k++)
					{
						r2 += r.data[k] * r.data[k];
					}
					const float r3 = r2 * std::sqrt(r2);

					for (int k = 0; k < 4; k++)
					{
						f[i].data[k] += r.data[k] / r3;
					}
				}
			}
		}

		#pragma omp for
		for (int i = 0; i < n; i++)
		{
			// a = f/m
			Vector4 a;
			for (int j = 0; j < 4; j++)
			{
				a.data[j] = f[i].data[j] * rm;
			}

			// x += v*dt + a*dt*dt/2
			for (int j = 0; j < 4; j++)
			{
				const float dxv = v[i].data[j] * dt;
				const float dxa = a.data[j] * tmp;
				const float dx = dxv + dxa;
				x[i].data[j] += dx;
			}

			// v += a*dt
			for (int j = 0; j < 4; j++)
			{
				const float dv = a.data[j] * dt;
				v[i].data[j] += dv;
			}
		}
	}
}
#endif

int main()
{
	const std::size_t n = 100;
	const int loop = 3;

	std::vector<utility::Vector4<float>> v(n);
    std::vector<utility::Vector4<float>> x(n);
    std::vector<utility::Vector4<float>> f(n);
	auto const generator = [](){return static_cast<float>(1 + std::rand()) / std::rand(); };
	auto const generator4 = [generator](){return utility::Vector4<float>(generator(), generator(), generator()); };
	boost::generate(v, generator4);
	boost::generate(x, generator4);

	Timer timer;

	const float dt = 0.1f;
	const float m = 2.5f;

	// なにもしない
	std::vector<utility::Vector4<float>> vNormal(n);
	std::vector<utility::Vector4<float>> xNormal(n);
	{
		boost::copy(v, vNormal.begin());
		boost::copy(x, xNormal.begin());

		std::cout << "Normal: ";
		timer.Start();
		for (auto i = 0; i < loop; i++)
		{
			Normal(xNormal, vNormal, f, m, dt, n);
		}
		const auto normalTime = timer.Time();
		std::cout << normalTime.count() << "[ms]" << std::endl;
	}

    using namespace boost;
    
	// OpenCL
    std::vector<utility::Vector4<float>> vOcl(n);
    std::vector<utility::Vector4<float>> xOcl(n);
	{
        boost::copy(v, vOcl.begin());
        boost::copy(x, xOcl.begin());

		std::cout << "OpenCL: ";

        compute::device device = compute::system::default_device();
        compute::context context(device);
        
        auto force_source = BOOST_COMPUTE_STRINGIZE_SOURCE(kernel void force(
            global float4 f[],
            const global float4 x[],
            const ulong n)
        {
            const int i = get_global_id(0);

            const float4 xi = x[i];

            float4 fi = 0;
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    const float4 xj = x[j];
                    const float4 r = xj - xi;

                    const float r2 = dot(r, r);
                    const float r3 = r2 * sqrt(r2);

                    fi += r / r3;
                }
            }

            f[i] = fi;
        });

        auto move_source = BOOST_COMPUTE_STRINGIZE_SOURCE(kernel void move(
            global float4 x[],
            global float4 v[],
            const global float4 f[],
            const float m, const float dt)
        {
            const int i = get_global_id(0);

            const float4 a = f[i] / m;

            const float4 xx = x[i];
            const float4 vv = v[i];

            x[i] = xx + vv*dt + a*dt*dt / 2;
            v[i] = vv + a*dt;
        });

		// プログラムの作成＆ビルド＆カーネルを作成
        auto kernel_force = compute::kernel::create_with_source(force_source, "force", context);
        auto kernel_move = compute::kernel::create_with_source(move_source, "move", context);

        compute::vector<compute::float4_> bufferX(n, context);
        compute::vector<compute::float4_> bufferV(n, context);
        compute::vector<compute::float4_> bufferF(n, context);

        kernel_force.set_args(
            bufferF,
		    bufferX,
		    static_cast<cl_ulong>(n));

		kernel_move.set_args(
            bufferX,
            bufferV,
            bufferF,
            static_cast<cl_float>(m),
            static_cast<cl_float>(dt));

		// キュー作成
        compute::command_queue queue(context, device);

		timer.Start();

        // ホスト->デバイス
        compute::copy(
            xOcl.begin(), xOcl.end(),
            bufferX.begin(),
            queue
            );

        compute::copy(
            vOcl.begin(), vOcl.end(),
            bufferV.begin(),
            queue
            );

		for (auto i = 0; i < loop; i++)
		{
			// 実行
            auto const event_force = queue.enqueue_1d_range_kernel(
                kernel_force,
                0,
                n,
                100);
            event_force.wait();

            auto const event_move = queue.enqueue_1d_range_kernel(
                kernel_move,
                0,
                n,
                100);
            event_move.wait();
		}
        
        std::vector<utility::Vector4<float>> x2(n);
        std::vector<utility::Vector4<float>> f2(n);

        compute::copy(
            bufferX.begin(), bufferX.end(),
            x2.begin(),
            queue
            );
        
        compute::copy(
            bufferF.begin(), bufferF.end(),
            f2.begin(),
            queue
            );

		const auto oclTime = timer.Time();
		std::cout << oclTime.count() << "[ms]" << std::endl;

		//std::cout << "== Platform : " << platform() << " ==" << std::endl;
		//std::cout <<
		//	"Name    : " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl <<
		//	"Vendor  : " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl <<
		//	"Version : " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;

		//std::cout << "== Device : " << device. << " ==" << std::endl;
		//std::cout <<
		//	"Name                                             : " << device.getInfo<CL_DEVICE_NAME>() << std::endl <<
		//	"Vendor                                           : " << device.getInfo<CL_DEVICE_VENDOR>() << " (ID:" << device.getInfo<CL_DEVICE_VENDOR_ID>() << ")" << std::endl <<
		//	"Version                                          : " << device.getInfo<CL_DEVICE_VERSION>() << std::endl <<
		//	"Driver version                                   : " << device.getInfo<CL_DRIVER_VERSION>() << std::endl <<
		//	"OpenCL C version                                 : " << device.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
	}

#if 0
	// エラーチェック
	const float eps = 1e-8;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			const float errorV = std::abs(vNormal[i].data[j] - vSimd[i].data[j]) / vNormal[i].data[j];
			if (errorV > eps)
			{
				std::cout << "error Simd V[" << i << "][" << j << "]: " << errorV << std::endl;
			}

			const float errorX = std::abs(xNormal[i].data[j] - xSimd[i].data[j]) / xNormal[i].data[j];
			if (errorV > eps)
			{
				std::cout << "error Simd X[" << i << "][" << j << "]" << errorX << std::endl;
			}
		}
	}
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			const float errorV = std::abs(vNormal[i].data[j] - vOcl[i].data[j]) / vNormal[i].data[j];
			if (errorV > eps)
			{
				std::cout << "error OpenCL V[" << i << "][" << j << "]: " << errorV << std::endl;
			}

			const float errorX = std::abs(xNormal[i].data[j] - xOcl[i].data[j]) / xNormal[i].data[j];
			if (errorV > eps)
			{
				std::cout << "error OpenCL X[" << i << "][" << j << "]" << errorX << std::endl;
			}
		}
	}
#endif

    getchar();
	return 0;
}
