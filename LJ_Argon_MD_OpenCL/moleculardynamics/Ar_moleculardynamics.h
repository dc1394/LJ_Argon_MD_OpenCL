/*! \file Ar_moleculardynamics.h
    \brief アルゴンに対して、分子動力学シミュレーションを行うクラスの宣言

    Copyright ©  2016 @dc1394 All Rights Reserved.
    This software is released under the BSD 2-Clause License.
*/

#ifndef _AR_MOLECULARDYNAMICS_H_
#define _AR_MOLECULARDYNAMICS_H_

#pragma once

#include "../myrandom/myrand.h"
#include "../myvector/vector4.h"
#include <cstdint>                                  // for std::int32_t
#include <cmath>                                    // for std::sqrt, std::pow
#include <iostream>                                 // for std::ios_base::fixed, std::ios_base::floatfield, std::cout
#include <vector>                                   // for std::vector
#include <boost/compute/algorithm/accumulate.hpp>   // for boost::compute::accumulate
#include <boost/compute/algorithm/fill.hpp>         // for boost::compute::fill
#include <boost/compute/algorithm/transform.hpp>    // for boost::compute::transform
#include <boost/compute/container/vector.hpp>       // for boost::compute::vector
#include <boost/compute/utility/source.hpp>         // for BOOST_COMPUTE_STRINGIZE_SOURCE
#include <boost/format.hpp>                         // for boost::format
#include <boost/optional.hpp>                       // for boost::optional
#include <boost/range/algorithm/generate.hpp>       // for boost::generate
#include <boost/utility/in_place_factory.hpp>       // for boost::in_place
#include <tbb/combinable.h>                         // for tbb::combinable
#include <tbb/parallel_for.h>                       // for tbb::parallel_for
#include <tbb/partitioner.h>                        // for tbb::auto_partitioner

namespace moleculardynamics {
    namespace compute = boost::compute;

    //! A class.
    /*!
        アルゴンに対して、分子動力学シミュレーションを行うクラス
    */
    template <typename T>
    class Ar_moleculardynamics final {
        // #region コンストラクタ・デストラクタ

    public:
        //! A constructor.
        /*!
            コンストラクタ
        */
        Ar_moleculardynamics();

        //! A destructor.
        /*!
            デフォルトデストラクタ
        */
        ~Ar_moleculardynamics() = default;

        // #endregion コンストラクタ・デストラクタ

        // #region publicメンバ関数

        //! A public member function.
        /*!
            原子に働く力を計算する
        */
        void Calc_Forces();

        //! A public member function.
        /*!
            原子に働く力を計算する
        */
        void Calc_Forces_OpenCL();

        //! A public member function.
        /*!
            OpenCLについての情報を表示する
        */
        void getinfo() const;

        //! A public member function.
        /*!
            原子を移動させる
        */
        void Move_Atoms();
        
        //! A public member function.
        /*!
            原子を移動させる
        */
        void Move_Atoms_OpenCL();
        
        //! A public member function.
        /*!
            初期化する
        */
        void reset();

        // #endregion publicメンバ関数

        // #region privateメンバ関数

    private:
        //! A private member function.
        /*!
            原子の初期位置を決める
        */
        void MD_initPos();

        //! A private member function.
        /*!
            原子の初期速度を決める
        */
        void MD_initVel();

        //! A private member function.
        /*!
            カーネルを設定する
        */
        void SetKernel();

        //! A private member function (constant).
        /*!
            ノルムの二乗を求める
            \param x x座標
            \param y y座標
            \param z z座標
            \return ノルムの二乗
        */
        T norm2(T x, T y, T z) const
        {
            return x * x + y * y + z * z;
        }

        // #endregion privateメンバ関数

        // #region メンバ変数

    public:
        //! A private member variable (constant).
        /*!
            初期のスーパーセルの個数
        */
        static auto constexpr FIRSTNC = 6;

        //! A private member variable (constant).
        /*!
            初期の格子定数のスケール
        */
        static T const FIRSTSCALE;

        //! A private member variable (constant).
        /*!
            初期温度（絶対温度）
        */
        static T const FIRSTTEMP;

    private:
        //! A private member variable (constant).
        /*!
            Woodcockの温度スケーリングの係数
        */
        static T const ALPHA;

        //! A private member variable (constant).
        /*!
            時間刻みΔt
        */
        static T const DT;
        
        //! A private member variable (constant).
        /*!
            ボルツマン定数
        */
        static T const KB;
        
        //! A private member variable (constant).
        /*!
            ローカルワークサイズ
        */
        static auto constexpr LOCALWORKSIZE = 96;

        //! A private member variable (constant).
        /*!
            アルゴン原子に対するε
        */
        static T const YPSILON;

        //! A private member variable (constant).
        /*!
            時間刻みの二乗
        */
        T const dt2;

        //! A private member variable.
        /*!
            OpenCLデバイス
        */
        compute::device device_;

        //! A private member variable.
        /*!
            OpenCL context
        */
        compute::context context_;

        //! A private member variable.
        /*!
            格子定数
        */
        T lat_;

        //! A private member variable (constant).
        /*!
            スーパーセルの個数
        */
        std::int32_t Nc_ = Ar_moleculardynamics::FIRSTNC;

        //! A private member variable.
        /*!
            n個目の原子に働く力
        */
        std::vector<myvector::Vector4<T>> F_;
        
        //! A private member variable.
        /*!
            n個目の原子に働く力（デバイス側）
        */
        compute::vector<compute::float4_> F_dev_;
        
        //! A private member variable.
        /*!
            周期境界条件をチェックするカーネル
        */
        compute::kernel kernel_check_periodic_;

        //! A private member variable.
        /*!
            各原子に働く力を計算するカーネル
        */
        compute::kernel kernel_force_;

        //! A private member variable.
        /*!
            Verlet法で時間発展するカーネル
        */
        compute::kernel kernel_move_atoms_;
        
        //! A private member variable.
        /*!
            修正Euler法で時間発展するカーネル
        */
        compute::kernel kernel_move_atoms1_;

        //! A private member variable.
        /*!
            MDのステップ数
        */
        std::int32_t MD_iter_;

        //! A private member variable (constant).
        /*!
            相互作用を計算するセルの個数
        */
        std::int32_t const ncp_ = 3;

        //! A private member variable.
        /*!
            原子数
        */
        std::int32_t NumAtom_;
        
        //! A private member variable.
        /*!
            周期境界条件の長さ
        */
        T periodiclen_;
        
        //! A private member variable.
        /*!
            ベクトルの大きさの二乗を求める関数オブジェクト
        */
        boost::optional<compute::function<float(compute::float4_)>> pnorm2_;

        //! A private member variable.
        /*!
            OpenCLのキュー
        */
        compute::command_queue queue_;

        //! A private member variable.
        /*!
            n個目の原子の座標
        */
        std::vector<myvector::Vector4<T>> r_;

        //! A private member variable.
        /*!
            n個目の原子の座標の複製
        */
        std::vector<myvector::Vector4<T>> r_clone_;

        //! A private member variable.
        /*!
            n個目の原子の座標（デバイス側）
        */
        compute::vector<compute::float4_> r_dev_;

        //! A private member variable.
        /*!
            n個目の原子の初期座標
        */
        std::vector<myvector::Vector4<T>> r1_;

        //! A private member variable.
        /*!
            n個目の原子の初期座標（デバイス側）
        */
        compute::vector<compute::float4_> r1_dev_;
        
        //! A private member variable (constant).
        /*!
            カットオフ半径
        */
        T const rc_ = 2.5;

        //! A private member variable (constant).
        /*!
            カットオフ半径の2乗
        */
        T const rc2_;

        //! A private member variable (constant).
        /*!
            カットオフ半径の逆数の6乗
        */
        T const rcm6_;

        //! A private member variable (constant).
        /*!
            カットオフ半径の逆数の12乗
        */
        T const rcm12_;

        //! A private member variable.
        /*!
            格子定数のスケーリングの定数
        */
        T scale_ = Ar_moleculardynamics::FIRSTSCALE;

        //! A private member variable.
        /*!
            計算された温度Tcalc
        */
        T Tc_;

        //! A private member variable.
        /*!
            与える温度Tgiven
        */
        T Tg_;
        
        //! A private member variable.
        /*!
            運動エネルギー
        */
        T Uk_;

        //! A private member variable.
        /*!
            ポテンシャルエネルギー
        */
        T Up_;

        //! A private member variable.
        /*!
            各原子のポテンシャルエネルギー（デバイス側）
        */
        compute::vector<float> Up_dev_;

        //! A private member variable.
        /*!
            全エネルギー
        */
        T Utot_;
        
        //! A private member variable.
        /*!
            n個目の原子の速度
        */
        std::vector<myvector::Vector4<T>> V_;

        //! A private member variable.
        /*!
            n個目の原子の速度（複製用）
        */
        std::vector<myvector::Vector4<T>> V_clone_;

        //! A private member variable.
        /*!
            n個目の原子の速度（デバイス側）
        */
        compute::vector<compute::float4_> V_dev_;

        //! A private member variable (constant).
        /*!
            ポテンシャルエネルギーの打ち切り
        */
        T const Vrc_;
        
        // #endregion メンバ変数

        // #region 禁止されたコンストラクタ・メンバ関数

        //! A private copy constructor (deleted).
        /*!
            コピーコンストラクタ（禁止）
        */
        Ar_moleculardynamics(Ar_moleculardynamics const &) = delete;

        //! A private member function (deleted).
        /*!
            operator=()の宣言（禁止）
            \param コピー元のオブジェクト（未使用）
            \return コピー元のオブジェクト
        */
        Ar_moleculardynamics & operator=(Ar_moleculardynamics const &) = delete;

        // #endregion 禁止されたコンストラクタ・メンバ関数
    };

    // #region static private 定数

    template <typename T>
    T const Ar_moleculardynamics<T>::FIRSTSCALE = 1.0;
    
    template <typename T>
    T const Ar_moleculardynamics<T>::FIRSTTEMP = 1.0;

    template <typename T>
    T const Ar_moleculardynamics<T>::ALPHA = 0.2;

    template <typename T>
    T const Ar_moleculardynamics<T>::DT = 0.001;

    template <typename T>
    T const Ar_moleculardynamics<T>::KB = 1.3806488E-23;

    template <typename T>
    T const Ar_moleculardynamics<T>::YPSILON = 1.6540172624E-21;

    // #endregion static private 定数

    // #region コンストラクタ

    template <typename T>
    Ar_moleculardynamics<T>::Ar_moleculardynamics()
        :
        device_(compute::system::default_device()),
        context_(device_),
        dt2(DT * DT),
        F_(Nc_ * Nc_ * Nc_ * 4),
        F_dev_(Nc_ * Nc_ * Nc_ * 4, context_),
        queue_(context_, device_),
        rc2_(rc_ * rc_),
        rcm6_(std::pow(rc_, -6.0)),
        rcm12_(std::pow(rc_, -12.0)),
        Tg_(Ar_moleculardynamics::FIRSTTEMP * Ar_moleculardynamics::KB / Ar_moleculardynamics::YPSILON),
        r_(Nc_ * Nc_ * Nc_ * 4),
        r_clone_(Nc_ * Nc_ * Nc_ * 4),
        r_dev_(Nc_ * Nc_ * Nc_ * 4, context_),
        r1_(Nc_ * Nc_ * Nc_ * 4),
        r1_dev_(Nc_ * Nc_ * Nc_ * 4, context_),
        Up_dev_(Nc_ * Nc_ * Nc_ * 4, context_),
        V_(Nc_ * Nc_ * Nc_ * 4),
        V_clone_(Nc_ * Nc_ * Nc_ * 4),
        V_dev_(Nc_ * Nc_ * Nc_ * 4, context_),
        Vrc_(4.0 * (rcm12_ - rcm6_))
    {
        // initalize parameters
        lat_ = std::pow(2.0, 2.0 / 3.0) * scale_;

        MD_iter_ = 1;

        MD_initPos();
        MD_initVel();

        periodiclen_ = lat_ * static_cast<T>(Nc_);

        SetKernel();
    }

    // #endregion コンストラクタ

    // #region publicメンバ関数

    template <typename T>
    void Ar_moleculardynamics<T>::Calc_Forces()
    {
        // 各原子に働く力の初期化
        for (auto n = 0; n < NumAtom_; n++) {
            F_[n].data[0] = static_cast<T>(0);
            F_[n].data[1] = static_cast<T>(0);
            F_[n].data[2] = static_cast<T>(0);
        }

        // ポテンシャルエネルギーの初期化
        tbb::combinable<T> Up;

        tbb::parallel_for(
            0,
            NumAtom_,
            1,
            [this, &Up](std::int32_t n) {
            for (auto m = 0; m < NumAtom_; m++) {

                // ±ncp_分のセル内の原子との相互作用を計算
                for (auto i = -ncp_; i <= ncp_; i++) {
                    for (auto j = -ncp_; j <= ncp_; j++) {
                        for (auto k = -ncp_; k <= ncp_; k++) {
                            auto const sx = static_cast<T>(i) * periodiclen_;
                            auto const sy = static_cast<T>(j) * periodiclen_;
                            auto const sz = static_cast<T>(k) * periodiclen_;

                            // 自分自身との相互作用を排除
                            if (n != m || i != 0 || j != 0 || k != 0) {
                                auto const dx = r_[n].data[0] - (r_[m].data[0] + sx);
                                auto const dy = r_[n].data[1] - (r_[m].data[1] + sy);
                                auto const dz = r_[n].data[2] - (r_[m].data[2] + sz);

                                auto const r2 = norm2(dx, dy, dz);
                                // 打ち切り距離内であれば計算
                                if (r2 <= rc2_) {
                                    auto const r = std::sqrt(r2);
                                    auto const rm6 = 1.0 / (r2 * r2 * r2);
                                    auto const rm7 = rm6 / r;
                                    auto const rm12 = rm6 * rm6;
                                    auto const rm13 = rm12 / r;

                                    auto const Fr = 48.0 * rm13 - 24.0 * rm7;

                                    F_[n].data[0] += dx / r * Fr;
                                    F_[n].data[1] += dy / r * Fr;
                                    F_[n].data[2] += dz / r * Fr;

                                    // エネルギーの計算、ただし二重計算のために0.5をかけておく
                                    Up.local() += 0.5 * (4.0 * (rm12 - rm6) - Vrc_);
                                }
                            }
                        }
                    }
                }
            }
        },
            tbb::auto_partitioner());
        
        Up_ = Up.combine(std::plus<T>());
    }

    template <typename T>
    void Ar_moleculardynamics<T>::Calc_Forces_OpenCL()
    {
        // ホスト→デバイス
        compute::copy(r_.begin(), r_.end(), r_dev_.begin(), queue_);

        compute::fill(F_dev_.begin(), F_dev_.end(), compute::float4_(0.0f), queue_);
        compute::fill(Up_dev_.begin(), Up_dev_.end(), 0.0f, queue_);
        
        // 各原子に働く力とポテンシャルエネルギーを計算
        auto const event_force = queue_.enqueue_1d_range_kernel(
            kernel_force_,
            0,
            NumAtom_,
            Ar_moleculardynamics::LOCALWORKSIZE);
        event_force.wait();

        // ポテンシャルエネルギーを計算
        Up_ = compute::accumulate(Up_dev_.begin(), Up_dev_.end(), 0.0f, queue_);
        
        // デバイス→ホスト
        compute::copy(F_dev_.begin(), F_dev_.end(), F_.begin(), queue_);
    }

    template <typename T>
    void Ar_moleculardynamics<T>::getinfo() const
    {
        std::cout << "== Platform : " << device_.platform().name() << " ==\n";
        std::cout <<
        	"Name    : " << device_.platform().get_info<CL_PLATFORM_NAME>() << '\n' <<
        	"Vendor  : " << device_.platform().get_info<CL_PLATFORM_VENDOR>() << '\n' <<
        	"Version : " << device_.platform().get_info<CL_PLATFORM_VERSION>() << '\n' ;

        std::cout << "== Device : " << device_.name() << " ==\n";
        std::cout <<
        	"Name               : " << device_.get_info<CL_DEVICE_NAME>() << '\n' <<
        	"Vendor             : " << device_.get_info<CL_DEVICE_VENDOR>() <<
            " (ID:" << device_.get_info<CL_DEVICE_VENDOR_ID>() << ")\n" <<
        	"Version            : " << device_.get_info<CL_DEVICE_VERSION>() << '\n' <<
        	"Driver version     : " << device_.get_info<CL_DRIVER_VERSION>() << '\n' <<
        	"OpenCL C version   : " << device_.get_info<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
    }

    template <typename T>
    void Ar_moleculardynamics<T>::Move_Atoms()
    {
        // 運動エネルギーの初期化
        Uk_ = 0.0;

        // 運動エネルギーの計算
        for (auto n = 0; n < NumAtom_; n++) {
            Uk_ += norm2(V_[n].data[0], V_[n].data[1], V_[n].data[2]);
        }
        Uk_ *= 0.5;

        // 全エネルギー（運動エネルギー+ポテンシャルエネルギー）の計算
        Utot_ = Uk_ + Up_;

        std::cout << boost::format("step = %d, 全エネルギー = %.8f, ポテンシャル = %.8f, 運動エネルギー = %.8f\n") % MD_iter_ % Utot_ % Up_ % Uk_;

        // 温度の計算
        Tc_ = Uk_ / (1.5 * static_cast<T>(NumAtom_));

        // calculate temperture
        auto const s = std::sqrt((Tg_ + Ar_moleculardynamics::ALPHA * (Tc_ - Tg_)) / Tc_);

        switch (MD_iter_) {
        case 1:
            // update the coordinates by the second order Euler method
            // 最初のステップだけ修正Euler法で時間発展
            tbb::parallel_for(
                0,
                NumAtom_,
                1,
                [this, s](std::int32_t n) {
                r1_[n].data = r_[n].data;

                // scaling of velocity
                V_[n].data[0] *= s;
                V_[n].data[1] *= s;
                V_[n].data[2] *= s;

                // update coordinates and velocity
                r_[n].data[0] += Ar_moleculardynamics::DT * V_[n].data[0] + 0.5 * F_[n].data[0] * dt2;
                r_[n].data[1] += Ar_moleculardynamics::DT * V_[n].data[1] + 0.5 * F_[n].data[1] * dt2;
                r_[n].data[2] += Ar_moleculardynamics::DT * V_[n].data[2] + 0.5 * F_[n].data[2] * dt2;

                V_[n].data[0] += Ar_moleculardynamics::DT * F_[n].data[0];
                V_[n].data[1] += Ar_moleculardynamics::DT * F_[n].data[1];
                V_[n].data[2] += Ar_moleculardynamics::DT * F_[n].data[2];
            },
                tbb::auto_partitioner());
        break;

        default:
            // update the coordinates by the Verlet method
            tbb::parallel_for(
                0,
                NumAtom_,
                1,
                [this, s](std::int32_t n) {
                    auto const rtmp = r_[n].data;
#ifdef NVE
                    r_[n].data[0] = 2.0 * r_[n].data[0] - r1_[n].data[0] + F_[n].data[0] * dt2;
                    r_[n].data[1] = 2.0 * r_[n].data[1] - r1_[n].data[1] + F_[n].data[1] * dt2;
                    r_[n].data[2] = 2.0 * r_[n].data[2] - r1_[n].data[2] + F_[n].data[2] * dt2;
#else
                    // update coordinates and velocity
                    // Verlet法の座標更新式において速度成分を抜き出し、その部分をスケールする
                    r_[n].data[0] += s * (r_[n].data[0] - r1_[n].data[0]) + F_[n].data[0] * dt2;
                    r_[n].data[1] += s * (r_[n].data[1] - r1_[n].data[1]) + F_[n].data[1] * dt2;
                    r_[n].data[2] += s * (r_[n].data[2] - r1_[n].data[2]) + F_[n].data[2] * dt2;
#endif

                    V_[n].data[0] = 0.5 * (r_[n].data[0] - r1_[n].data[0]) / Ar_moleculardynamics::DT;
                    V_[n].data[1] = 0.5 * (r_[n].data[1] - r1_[n].data[1]) / Ar_moleculardynamics::DT;
                    V_[n].data[2] = 0.5 * (r_[n].data[2] - r1_[n].data[2]) / Ar_moleculardynamics::DT;
            
                    r1_[n].data = rtmp;
                },
                tbb::auto_partitioner());
            break;
        }

        // consider the periodic boundary condination
        // セルの外側に出たら座標をセル内に戻す
        tbb::parallel_for(
            0,
            NumAtom_,
            1,
            [this](std::int32_t n) {
            for (auto i = 0; i < 3; i++) {
                if (r_[n].data[i] > periodiclen_) {
                    r_[n].data[i] -= periodiclen_;
                    r1_[n].data[i] -= periodiclen_;
                }
                else if (r_[n].data[i] < 0.0) {
                    r_[n].data[i] += periodiclen_;
                    r1_[n].data[i] += periodiclen_;
                }
            }
        },
            tbb::auto_partitioner());

        MD_iter_++;
    }

    template <typename T>
    void Ar_moleculardynamics<T>::Move_Atoms_OpenCL()
    {
        // ホスト→デバイス
        compute::copy(r_.begin(), r_.end(), r_dev_.begin(), queue_ );
        compute::copy(r1_.begin(), r1_.end(), r1_dev_.begin(), queue_ );
        compute::copy(V_.begin(), V_.end(), V_dev_.begin(), queue_);

        // 運動エネルギーの計算
        compute::vector<float> V2_dev_(NumAtom_, context_);
        compute::transform(V_dev_.begin(), V_dev_.end(), V2_dev_.begin(), *pnorm2_, queue_);
        Uk_ = compute::accumulate(V2_dev_.begin(), V2_dev_.end(), 0.0f, queue_) * 0.5;

        // 全エネルギー（運動エネルギー+ポテンシャルエネルギー）の計算
        Utot_ = Uk_ + Up_;

        std::cout << boost::format("step = %d, 全エネルギー = %.8f, ポテンシャル = %.8f, 運動エネルギー = %.8f\n") % MD_iter_ % Utot_ % Up_ % Uk_;

        // 温度の計算
        Tc_ = Uk_ / (1.5 * static_cast<T>(NumAtom_));

        // calculate temperture
        auto const s = std::sqrt((Tg_ + Ar_moleculardynamics::ALPHA * (Tc_ - Tg_)) / Tc_);

        switch (MD_iter_) {
        case 1:
            {
                // update the coordinates by the second order Euler method
                // 最初のステップだけ修正Euler法で時間発展
                kernel_move_atoms1_.set_args(
                    r_dev_,
                    r1_dev_,
                    V_dev_,
                    F_dev_,
                    Ar_moleculardynamics::DT,
                    s);

                auto const event_move_atoms1 = queue_.enqueue_1d_range_kernel(
                    kernel_move_atoms1_,
                    0,
                    NumAtom_,
                    Ar_moleculardynamics::LOCALWORKSIZE);
                event_move_atoms1.wait();
            }
        break;

        default:
            {
                // update the coordinates by the Verlet method
                kernel_move_atoms_.set_args(
                    r_dev_,
                    r1_dev_,
                    V_dev_,
                    F_dev_,
                    Ar_moleculardynamics::DT,
                    s);

                auto const event_move_atoms = queue_.enqueue_1d_range_kernel(
                    kernel_move_atoms_,
                    0,
                    NumAtom_,
                    Ar_moleculardynamics::LOCALWORKSIZE);
                event_move_atoms.wait();
            }
        break;
        }

        // 周期境界条件のチェック
        auto const event_check_periodic = queue_.enqueue_1d_range_kernel(
            kernel_check_periodic_,
            0,
            NumAtom_,
            Ar_moleculardynamics::LOCALWORKSIZE);
        event_check_periodic.wait();

        // デバイス→ホスト
        compute::copy(r_dev_.begin(), r_dev_.end(), r_.begin(), queue_);
        compute::copy(r1_dev_.begin(), r1_dev_.end(), r1_.begin(), queue_);
        compute::copy(V_dev_.begin(), V_dev_.end(), V_.begin(), queue_);

        MD_iter_++;
    }

    template <typename T>
    void Ar_moleculardynamics<T>::reset()
    {
        MD_iter_ = 1;
        
        r_.assign(r_clone_.begin(), r_clone_.end());
        V_.assign(V_clone_.begin(), V_clone_.end());
    }

    // #endregion publicメンバ関数

    // #region privateメンバ関数

    template <typename T>
    void Ar_moleculardynamics<T>::MD_initPos()
    {
        T sx, sy, sz;
        auto n = 0;

        for (auto i = 0; i < Nc_; i++) {
            for (auto j = 0; j < Nc_; j++) {
                for (auto k = 0; k < Nc_; k++) {
                    // 基本セルをコピーする
                    sx = static_cast<T>(i) * lat_;
                    sy = static_cast<T>(j) * lat_;
                    sz = static_cast<T>(k) * lat_;

                    // 基本セル内には4つの原子がある
                    r_[n].data[0] = sx;
                    r_[n].data[1] = sy;
                    r_[n].data[2] = sz;
                    n++;

                    r_[n].data[0] = 0.5 * lat_ + sx;
                    r_[n].data[1] = 0.5 * lat_ + sy;
                    r_[n].data[2] = sz;
                    n++;

                    r_[n].data[0] = sx;
                    r_[n].data[1] = 0.5 * lat_ + sy;
                    r_[n].data[2] = 0.5 * lat_ + sz;
                    n++;

                    r_[n].data[0] = 0.5 * lat_ + sx;
                    r_[n].data[1] = sy;
                    r_[n].data[2] = 0.5 * lat_ + sz;
                    n++;
                }
            }
        }

        NumAtom_ = n;

        // move the center of mass to the origin
        // 系の重心を座標系の原点とする
        sx = 0.0;
        sy = 0.0;
        sz = 0.0;

        for (auto n = 0; n < NumAtom_; n++) {
            sx += r_[n].data[0];
            sy += r_[n].data[1];
            sz += r_[n].data[2];
        }

        sx /= static_cast<T>(NumAtom_);
        sy /= static_cast<T>(NumAtom_);
        sz /= static_cast<T>(NumAtom_);

        for (auto n = 0; n < NumAtom_; n++) {
            r_[n].data[0] -= sx;
            r_[n].data[1] -= sy;
            r_[n].data[2] -= sz;
        }

        // 原子座標を複製
        r_clone_.assign(r_.begin(), r_.end());
    }

    template <typename T>
    void Ar_moleculardynamics<T>::MD_initVel()
    {
        auto const v = std::sqrt(3.0 * Tg_);

        myrandom::MyRand mr(-1.0, 1.0);

        auto const generator4 = [this, &mr, v]() {
            T rndX = mr.myrand();
            T rndY = mr.myrand();
            T rndZ = mr.myrand();
            T const tmp = 1.0 / std::sqrt(norm2(rndX, rndY, rndZ));
            rndX *= tmp;
            rndY *= tmp;
            rndZ *= tmp;
            
            // 方向はランダムに与える
            return myvector::Vector4<T>(v * rndX, v * rndY, v * rndZ);
        };

        boost::generate(V_, generator4);

        auto sx = 0.0;
        auto sy = 0.0;
        auto sz = 0.0;

        for (auto n = 0; n < NumAtom_; n++) {
            sx += V_[n].data[0];
            sy += V_[n].data[1];
            sz += V_[n].data[2];
        }

        sx /= static_cast<T>(NumAtom_);
        sy /= static_cast<T>(NumAtom_);
        sz /= static_cast<T>(NumAtom_);

        // 重心の並進運動を避けるために、速度の和がゼロになるように補正
        for (auto n = 0; n < NumAtom_; n++) {
            V_[n].data[0] -= sx;
            V_[n].data[1] -= sy;
            V_[n].data[2] -= sz;
        }

        // 速度の可変長配列を複製
        V_clone_.assign(V_.begin(), V_.end());
    }

    template <typename T>  
    void Ar_moleculardynamics<T>::SetKernel()
    {
        using namespace boost::compute;

        auto const check_periodic_source = BOOST_COMPUTE_STRINGIZE_SOURCE(kernel void check_periodic(
            __global float4 r[],
            __global float4 r1[],
            __const float periodiclen)
        {
            int const n = get_global_id(0);

            if (r[n].x > periodiclen) {
                r[n].x -= periodiclen;
                r1[n].x -= periodiclen;
            }
            else if (r[n].x < 0.0f) {
                r[n].x += periodiclen;
                r1[n].x += periodiclen;
            }

            if (r[n].y > periodiclen) {
                r[n].y -= periodiclen;
                r1[n].y -= periodiclen;
            }
            else if (r[n].y < 0.0f) {
                r[n].y += periodiclen;
                r1[n].y += periodiclen;
            }

            if (r[n].z > periodiclen) {
                r[n].z -= periodiclen;
                r1[n].z -= periodiclen;
            }
            else if (r[n].z < 0.0f) {
                r[n].z += periodiclen;
                r1[n].z += periodiclen;
            }
        });

        kernel_check_periodic_ = kernel::create_with_source(check_periodic_source, "check_periodic", context_);
        kernel_check_periodic_.set_args(r_dev_, r1_dev_, periodiclen_);

        auto const force_source = BOOST_COMPUTE_STRINGIZE_SOURCE(kernel void force(
            __global float4 f[],
            __global float Up[],
            __global __const float4 rv[],
            __const int ncp,
            __const int numatom,
            __const float periodiclen,
            __const float rc2,
            __const float Vrc)
        {
            int const n = get_global_id(0);

            for (int m = 0; m < numatom; m++) {

                // ±ncp分のセル内の原子との相互作用を計算
                for (int i = -ncp; i <= ncp; i++) {
                    for (int j = -ncp; j <= ncp; j++) {
                        for (int k = -ncp; k <= ncp; k++) {
                            float4 s;
                            s.x = (float)(i) * periodiclen;
                            s.y = (float)(j) * periodiclen;
                            s.z = (float)(k) * periodiclen;
                            s.w = 0.0f;

                            // 自分自身との相互作用を排除
                            if (n != m || i != 0 || j != 0 || k != 0) {
                                float4 const d = rv[n] - (rv[m] + s);

                                float const r2 = dot(d, d);
                                // 打ち切り距離内であれば計算
                                if (r2 <= rc2) {
                                    float const r = sqrt(r2);
                                    float const rm6 = 1.0 / (r2 * r2 * r2);
                                    float const rm7 = rm6 / r;
                                    float const rm12 = rm6 * rm6;
                                    float const rm13 = rm12 / r;

                                    float const Fr = 48.0 * rm13 - 24.0 * rm7;

                                    f[n] += d / (float4)(r) * (float4)(Fr);
                                    Up[n] += 0.5 * (4.0 * (rm12 - rm6) - Vrc);
                                }
                            }
                        }
                    }
                }
            }
        });

        kernel_force_ = kernel::create_with_source(force_source, "force", context_);
        kernel_force_.set_args(
            F_dev_,
            Up_dev_,
            r_dev_,
            ncp_,
            NumAtom_,
            periodiclen_,
            rc2_,
            Vrc_);

        auto const move_atoms_source = BOOST_COMPUTE_STRINGIZE_SOURCE(kernel void move_atoms(
            __global float4 r[],
            __global float4 r1[],
            __global float4 V[],
            __global __const float4 F[],
            __const float deltat,
            __const float s)
        {
            int const n = get_global_id(0);
            float4 const dt = (float4)(deltat);
            float4 const dt2 = dt * dt;
            float4 const rtmp = r[n];
#ifdef NVE
            r[n] = (float4)(2.0f) * r[n] - r1[n] + F[n] * dt2;
#else
            // update coordinates and velocity
            // Verlet法の座標更新式において速度成分を抜き出し、その部分をスケールする
            r[n] += (float4)(s)* (r[n] - r1[n]) + F[n] * dt2;
#endif
            V[n] = (float4)(0.5f) * (r[n] - r1[n]) / dt;

            r1[n] = rtmp;
        });

        kernel_move_atoms_ = kernel::create_with_source(move_atoms_source, "move_atoms", context_);

        auto const move_atoms1_source = BOOST_COMPUTE_STRINGIZE_SOURCE(kernel void move_atoms1(
            __global float4 r[],
            __global float4 r1[],
            __global float4 V[],
            __global __const float4 F[],
            __const float deltat,
            __const float s)
        {
            int const n = get_global_id(0);
            float4 const dt = (float4)(deltat);
            float4 const dt2 = dt * dt;

            r1[n] = r[n];

            // scaling of velocity
            V[n] *= (float4)(s);

            // update coordinates and velocity
            r[n] += dt * V[n] + (float4)(0.5f) * F[n] * dt2;

            V[n] += (float4)(dt) * F[n];
        });

        kernel_move_atoms1_ = kernel::create_with_source(move_atoms1_source, "move_atoms1", context_);

        pnorm2_ = boost::in_place(make_function_from_source<float(float4_)>(
            "norm2",
            "float norm2(float4 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }"));
    }

    // #endregion privateメンバ関数
}

#endif      // _AR_MOLECULARDYNAMICS_H_
