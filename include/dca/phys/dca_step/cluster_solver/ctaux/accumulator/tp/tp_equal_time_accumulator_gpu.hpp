// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Peter Staar (taa@zurich.ibm.com)
//         Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// This class measures the equal time operator functions.

#ifndef DCA_HAVE_CUDA
#error "This file requires CUDA."
#endif

#ifndef DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_CTAUX_ACCUMULATOR_TP_TP_EQUAL_TIME_ACCUMULATOR_GPU_HPP
#define DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_CTAUX_ACCUMULATOR_TP_TP_EQUAL_TIME_ACCUMULATOR_GPU_HPP

#include "dca/phys/dca_step/cluster_solver/ctaux/accumulator/tp/tp_equal_time_accumulator.hpp"
//#include "dca/phys/dca_step/cluster_solver/ctaux/accumulator/tp/singleton_obj_dev.hpp"

#include <cuda.h>
#include <mutex>

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dca/linalg/lapack/magma.hpp"
#include "dca/linalg/util/magma_queue.hpp"

#include "dca/linalg/matrix.hpp"
#include "dca/linalg/util/allocators/vectors_typedefs.hpp"
#include "dca/linalg/util/cuda_event.hpp"
#include "dca/linalg/vector.hpp"

#include "dca/function/domains.hpp"
#include "dca/function/function.hpp"
#include "dca/math/function_transform/function_transform.hpp"
#include "dca/math/interpolation/akima_interpolation.hpp"
#include "dca/phys/dca_step/cluster_solver/ctaux/structs/vertex_singleton.hpp"
#include "dca/phys/domains/cluster/cluster_domain.hpp"
#include "dca/phys/domains/cluster/cluster_domain_aliases.hpp"
#include "dca/phys/domains/quantum/electron_band_domain.hpp"
#include "dca/phys/domains/quantum/electron_spin_domain.hpp"
#include "dca/phys/domains/time_and_frequency/time_domain.hpp"
#include "dca/phys/domains/time_and_frequency/time_domain_left_oriented.hpp"
#include "dca/phys/domains/time_and_frequency/vertex_time_domain.hpp"
#include "dca/phys/dca_step/cluster_solver/ctaux/accumulator/tp/kernels_interface_for_eqtime.hpp"
#include "dca/phys/dca_step/cluster_solver/ctaux/accumulator/tp/TpEqTime_helper.cuh"
#include "dca/util/plot.hpp"

namespace dca {
namespace phys {
namespace solver {
namespace ctaux {
// dca::phys::solver::ctaux::

template <class parameters_type, class MOMS_type>
class TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::GPU> : public TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::CPU> {
private:
  using this_type = TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::GPU>;
  using BaseClass = TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::CPU>;


public:
  typedef double scalar_type;
  typedef vertex_singleton vertex_singleton_type;

  typedef typename parameters_type::profiler_type profiler_type;

  using t =  typename BaseClass::t;
  using t_VERTEX = typename BaseClass::t_VERTEX;

  using b = typename BaseClass::b;
  using s = typename BaseClass::s;
  using nu = typename BaseClass::nu;  // orbital-spin index

  using cCDA = typename BaseClass::CDA;
  using RClusterDmn = typename BaseClass::RClusterDmn;
  using KClusterDmn = typename BaseClass::KClusterDmn;
  using r_dmn_t = typename BaseClass::r_dmn_t;
  using k_dmn_t = typename BaseClass::k_dmn_t;

  using b_r_t_VERTEX_dmn_t = typename BaseClass::b_r_t_VERTEX_dmn_t;

  using shifted_t = typename BaseClass::shifted_t;
  using nu_nu_r_dmn_t_shifted_t = typename BaseClass::nu_nu_r_dmn_t_shifted_t;

  using akima_dmn_t = typename BaseClass::akima_dmn_t;
  using akima_nu_nu_r_dmn_t_shifted_t = typename BaseClass::akima_nu_nu_r_dmn_t_shifted_t;

public:
  TpEqualTimeAccumulator( parameters_type& parameters_ref, MOMS_type& MOMS_ref, int id);

  void resetAccumulation();

  void finalize();

  void sumTo(TpEqualTimeAccumulator<parameters_type, MOMS_type,linalg::GPU>& other);

  
  auto get_stream() const {
    return streams_[0];
  }

  void synchronizeCopy() {
  event_.block(streams_[0]);
  event_.block(streams_[1]);
  }

  void syncStreams(const linalg::util::CudaEvent& event) {
    for (const auto& stream : streams_)
      event.block(stream);
  }

  void synchronizeStreams();

  template <class configuration_type, typename RealInp>
  void compute_G_r_t(const configuration_type& configuration_e_up,
                     const dca::linalg::Matrix<RealInp, dca::linalg::GPU>& M_up,
                     const configuration_type& configuration_e_dn,
                     const dca::linalg::Matrix<RealInp, dca::linalg::GPU>& M_dn);

  void accumulate_G_r_t(double sign);
  void accumulate_G_r_t_orig(double sign);

  void accumulate_moments(double sign);

  void accumulate_dwave_pp_correlator(double sign);

  // Accumulate all relevant quantities. This is equivalent to calling compute_G_r_t followed by all
  // the accumulation methods.
  template <class configuration_type, typename RealInp>
  void accumulateAll(const configuration_type& configuration_e_up,
                     const dca::linalg::Matrix<RealInp, dca::linalg::GPU>& M_up,
                     const configuration_type& configuration_e_dn,
                     const dca::linalg::Matrix<RealInp, dca::linalg::GPU>& M_dn, int sign);

  template <class configuration_type, typename RealInp>
  void accumulateAll(const configuration_type& configuration_e_up,
                     const dca::linalg::Matrix<RealInp, dca::linalg::CPU>& M_up,
                     const configuration_type& configuration_e_dn,
                     const dca::linalg::Matrix<RealInp, dca::linalg::CPU>& M_dn, int sign);

  double get_GFLOP();


  func::function<double, func::dmn_variadic<nu, nu, r_dmn_t, t>>& get_G_r_t() {
    return G_r_t;
  }
  func::function<double, func::dmn_variadic<nu, nu, r_dmn_t, t>>& get_G_r_t_stddev() {
    return G_r_t_stddev;
  }
  func::function<double, func::dmn_variadic<b, r_dmn_t>>& get_charge_cluster_moment() {
    return charge_cluster_moment;
  }
  func::function<double, func::dmn_variadic<b, r_dmn_t>>& get_magnetic_cluster_moment() {
    return magnetic_cluster_moment;
  }
  func::function<double, func::dmn_variadic<b, r_dmn_t>>& get_dwave_pp_correlator() {
    return dwave_pp_correlator;
  }
  func::function<double, func::dmn_variadic<b, b, r_dmn_t, t_VERTEX>>& get_spin_ZZ_chi() {
    return spin_ZZ_chi_accumulated;
  }
  func::function<double, func::dmn_variadic<b, b, r_dmn_t, t_VERTEX>>& get_spin_ZZ_chi_stddev() {
    return spin_ZZ_chi_stddev;
  }
  func::function<double, func::dmn_variadic<b, b, r_dmn_t, t_VERTEX>>& get_spin_XX_chi() {
    return spin_XX_chi_accumulated;
  }

private:

  void initialize_my_configuration_ondevice();
  void initialize_TpEqTime_helper() ;

  int find_first_non_interacting_spin(const std::vector<vertex_singleton_type>& configuration_e_spin);

  template <class configuration_type>
  void compute_G0_matrix(e_spin_states e_spin, const configuration_type& configuration,
                         dca::linalg::Matrix<float, dca::linalg::GPU>& G0_matrix);

  template <class configuration_type>
  void compute_G0_matrix_left(e_spin_states e_spin, const configuration_type& configuration,
                              dca::linalg::Matrix<float, dca::linalg::GPU>& G0_matrix);

  template <class configuration_type>
  void compute_G0_matrix_right(e_spin_states e_spin, const configuration_type& configuration,
                               dca::linalg::Matrix<float, dca::linalg::GPU>& G0_matrix);

  double interpolate_akima(int b_i, int s_i, int b_j, int s_j, int delta_r, double tau);

 
  void interpolate(func::function<double, func::dmn_variadic<nu, nu, r_dmn_t, t>>& G_r_t,
                   func::function<double, func::dmn_variadic<nu, nu, r_dmn_t, t>>& G_r_t_stddev);

private:

  double beta_;
  double sp_time_intervals_;

  double GFLOP;
 
  int dwave_config_size;
  constexpr static int n_bands_ = parameters_type::model_type::BANDS;
 
  std::array<int, 2> streams_id_;
  std::array<cudaStream_t, 2> streams_;
  linalg::util::CudaEvent event_;

  using BaseClass::measureNow;

  using BaseClass::thread_id;
  using BaseClass::b_r_t_dmn;
  using BaseClass::nu_nu_r_dmn_t_t_shifted_dmn;
  using BaseClass::akima_coefficients;

  using BaseClass::fixed_configuration;

  using BaseClass::G0_sign_up;
  using BaseClass::G0_sign_dn;

  using BaseClass::G0_indices_up;
  using BaseClass::G0_indices_dn;

  using BaseClass::G0_integration_factor_up;
  using BaseClass::G0_integration_factor_dn;

  using BaseClass::G0_original_up;
  using BaseClass::G0_original_dn;

  using BaseClass::M_matrix_up;
  using BaseClass::M_matrix_dn;

  using BaseClass::G0_matrix_up;
  using BaseClass::G0_matrix_dn;

  using BaseClass::G0_matrix_up_left;
  using BaseClass::G0_matrix_dn_left;

  using BaseClass::G0_matrix_up_right;
  using BaseClass::G0_matrix_dn_right;

  using BaseClass::M_G0_matrix_up;
  using BaseClass::M_G0_matrix_dn;

  using BaseClass::G0_M_G0_matrix_up;
  using BaseClass::G0_M_G0_matrix_dn;

  using BaseClass::G_r_t_dn;
  using BaseClass::G_r_t_up;

  using BaseClass::G_r_t;
  using BaseClass::G_r_t_stddev;

  using BaseClass::G_r_t_accumulated;
  using BaseClass::G_r_t_accumulated_squared;


  using BaseClass::spin_ZZ_chi_accumulated;
  using BaseClass::spin_ZZ_chi_stddev;
  using BaseClass::spin_XX_chi_accumulated;

  using BaseClass::charge_cluster_moment;
  using BaseClass::magnetic_cluster_moment;

  using BaseClass::dwave_k_factor;
  using BaseClass::dwave_r_factor;

  using BaseClass::dwave_pp_correlator;

  using MatrixHost_dble = dca::linalg::Matrix<double, dca::linalg::CPU>;
  using MatrixHost_flt = dca::linalg::Matrix<float, dca::linalg::CPU>;
  using MatrixHost_int = dca::linalg::Matrix<int, dca::linalg::CPU>;
  using VectorHost_dble = dca::linalg::Vector<double, dca::linalg::CPU>;
  using VectorHost_int = dca::linalg::Vector<int, dca::linalg::CPU>;

  dca::linalg::Matrix<float, dca::linalg::GPU> G0_matrix_up_left_dev;
  dca::linalg::Matrix<float, dca::linalg::GPU> G0_matrix_dn_left_dev;

  dca::linalg::Matrix<float, dca::linalg::GPU> G0_matrix_up_right_dev;
  dca::linalg::Matrix<float, dca::linalg::GPU> G0_matrix_dn_right_dev;

  dca::linalg::Matrix<float, dca::linalg::GPU> M_matrix_up_dev;
  dca::linalg::Matrix<float, dca::linalg::GPU> M_matrix_dn_dev;

  dca::linalg::Matrix<float, dca::linalg::GPU> M_G0_matrix_up_dev;
  dca::linalg::Matrix<float, dca::linalg::GPU> M_G0_matrix_dn_dev;
  
  dca::linalg::Matrix<float, dca::linalg::GPU> G0_M_G0_matrix_up_dev;
  dca::linalg::Matrix<float, dca::linalg::GPU> G0_M_G0_matrix_dn_dev;

  dca::linalg::Matrix<float, dca::linalg::GPU> G_r_t_up_dev;
  dca::linalg::Matrix<float, dca::linalg::GPU> G_r_t_dn_dev;
 
  dca::linalg::Vector<double, dca::linalg::GPU> G_r_t_accumulated_dev;
  dca::linalg::Vector<double, dca::linalg::GPU> G_r_t_accumulated_squared_dev;


  dca::linalg::Vector<double, dca::linalg::GPU> spin_ZZ_chi_accumulated_dev;
  dca::linalg::Vector<double, dca::linalg::GPU> spin_ZZ_chi_stddev_dev;
  dca::linalg::Vector<double, dca::linalg::GPU> spin_XX_chi_accumulated_dev;
  dca::linalg::Vector<double, dca::linalg::GPU> dwave_pp_correlator_dev;
 



  dca::linalg::util::HostVector<ConfigElemTpEqTime> config_up_;
  dca::linalg::util::HostVector<ConfigElemTpEqTime> config_dn_;

  dca::linalg::Vector<ConfigElemTpEqTime, linalg::GPU> config_up_dev_;
  dca::linalg::Vector<ConfigElemTpEqTime, linalg::GPU> config_dn_dev_;
  

};

template <class parameters_type, class MOMS_type>
TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::GPU>::TpEqualTimeAccumulator(
     parameters_type& parameters_ref, MOMS_type& MOMS_ref, int id)
    : BaseClass(parameters_ref, MOMS_ref, id),

      beta_(parameters_ref.get_beta()),

      sp_time_intervals_(parameters_ref.get_sp_time_intervals()),

      GFLOP(0),

      streams_id_{0,1},

      streams_(){

  G_r_t_up_dev.resizeNoCopy(std::make_pair(b_r_t_VERTEX_dmn_t::dmn_size(), b_r_t_VERTEX_dmn_t::dmn_size()));
  assert(cudaPeekAtLastError() == cudaSuccess);
  G_r_t_dn_dev.resizeNoCopy(std::make_pair(b_r_t_VERTEX_dmn_t::dmn_size(), b_r_t_VERTEX_dmn_t::dmn_size()));
  assert(cudaPeekAtLastError() == cudaSuccess);
  G_r_t_accumulated_dev.resizeNoCopy(nu::dmn_size()*nu::dmn_size()*r_dmn_t::dmn_size()*t_VERTEX::dmn_size());
  assert(cudaPeekAtLastError() == cudaSuccess);
  G_r_t_accumulated_squared_dev.resizeNoCopy(nu::dmn_size()*nu::dmn_size()*r_dmn_t::dmn_size()*t_VERTEX::dmn_size());
  assert(cudaPeekAtLastError() == cudaSuccess);


  spin_ZZ_chi_accumulated_dev.resizeNoCopy(b::dmn_size()*b::dmn_size()*r_dmn_t::dmn_size()*t_VERTEX::dmn_size());
  assert(cudaPeekAtLastError() == cudaSuccess);
  spin_ZZ_chi_stddev_dev.resizeNoCopy(b::dmn_size()*b::dmn_size()*r_dmn_t::dmn_size()*t_VERTEX::dmn_size());
  assert(cudaPeekAtLastError() == cudaSuccess);
  spin_XX_chi_accumulated_dev.resizeNoCopy(b::dmn_size()*b::dmn_size()*r_dmn_t::dmn_size()*t_VERTEX::dmn_size());
  assert(cudaPeekAtLastError() == cudaSuccess);
  dwave_pp_correlator_dev.resizeNoCopy(b::dmn_size()*r_dmn_t::dmn_size());
  assert(cudaPeekAtLastError() == cudaSuccess);


}


template <class parameters_type, class MOMS_type>
double TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::GPU>::get_GFLOP() {
  double tmp = GFLOP;
  GFLOP = 0;
  return tmp;
}

template <class parameters_type, class MOMS_type>
void TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::GPU>::initialize_my_configuration_ondevice()  {
}


template <class parameters_type, class MOMS_type>
void TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::GPU>::initialize_TpEqTime_helper()  {

        
 dwave_config_size=b::dmn_size() * b::dmn_size() * b::dmn_size() * r_dmn_t::dmn_size() * r_dmn_t::dmn_size() * r_dmn_t::dmn_size();

  static std::once_flag flag;
  std::call_once(flag, [&]() {



	 VectorHost_int fixed_config_b_ind;
	 VectorHost_int fixed_config_r_ind;
	 VectorHost_int fixed_config_t_ind;
	 VectorHost_dble fixed_config_t_val;

         fixed_config_b_ind.resizeNoCopy(b::dmn_size() * r_dmn_t::dmn_size() * t_VERTEX::dmn_size());
         fixed_config_r_ind.resizeNoCopy(b::dmn_size() * r_dmn_t::dmn_size() * t_VERTEX::dmn_size());
         fixed_config_t_ind.resizeNoCopy(b::dmn_size() * r_dmn_t::dmn_size() * t_VERTEX::dmn_size());
         fixed_config_t_val.resizeNoCopy(b::dmn_size() * r_dmn_t::dmn_size() * t_VERTEX::dmn_size());

  int index = 0;
  for (int b_ind = 0; b_ind < b::dmn_size(); b_ind++) {
    for (int r_ind = 0; r_ind < r_dmn_t::dmn_size(); r_ind++) {
      for (int t_ind = 0; t_ind < t_VERTEX::dmn_size(); t_ind++) {
        
	fixed_config_b_ind[index] = fixed_configuration[index].b_ind;
	fixed_config_r_ind[index] = fixed_configuration[index].r_ind;
	fixed_config_t_ind[index] = fixed_configuration[index].t_ind;
	fixed_config_t_val[index] = fixed_configuration[index].t_val;

        index += 1;
      }
    }
  }
	//copy for d-wave calculations
	
	 VectorHost_int dwave_config_r_i;
	 VectorHost_int dwave_config_r_j;
	 VectorHost_int dwave_config_r_l;
	 VectorHost_int dwave_config_b_i;
	 VectorHost_int dwave_config_b_j;
	 VectorHost_int dwave_config_b_l;

         dwave_config_r_i.resizeNoCopy(b::dmn_size() * b::dmn_size() * b::dmn_size() * r_dmn_t::dmn_size() * r_dmn_t::dmn_size() * r_dmn_t::dmn_size());
         dwave_config_r_j.resizeNoCopy(b::dmn_size() * b::dmn_size() * b::dmn_size() * r_dmn_t::dmn_size() * r_dmn_t::dmn_size() * r_dmn_t::dmn_size());
         dwave_config_r_l.resizeNoCopy(b::dmn_size() * b::dmn_size() * b::dmn_size() * r_dmn_t::dmn_size() * r_dmn_t::dmn_size() * r_dmn_t::dmn_size());
         dwave_config_b_i.resizeNoCopy(b::dmn_size() * b::dmn_size() * b::dmn_size() * r_dmn_t::dmn_size() * r_dmn_t::dmn_size() * r_dmn_t::dmn_size());
         dwave_config_b_j.resizeNoCopy(b::dmn_size() * b::dmn_size() * b::dmn_size() * r_dmn_t::dmn_size() * r_dmn_t::dmn_size() * r_dmn_t::dmn_size());
         dwave_config_b_l.resizeNoCopy(b::dmn_size() * b::dmn_size() * b::dmn_size() * r_dmn_t::dmn_size() * r_dmn_t::dmn_size() * r_dmn_t::dmn_size());


index = 0;
  for (int r_i = 0; r_i < r_dmn_t::dmn_size(); r_i++) {
    for (int r_j = 0; r_j < r_dmn_t::dmn_size(); r_j++) {
      for (int r_l = 0; r_l < r_dmn_t::dmn_size(); r_l++) {
          for (int b_i = 0; b_i < b::dmn_size(); b_i++) {
            for (int b_j = 0; b_j < b::dmn_size(); b_j++) {
              for (int b_l = 0; b_l < b::dmn_size(); b_l++) {
		
		dwave_config_r_i[index] = r_i;
		dwave_config_r_j[index] = r_j;
		dwave_config_r_l[index] = r_l;
		dwave_config_b_i[index] = b_i;
		dwave_config_b_j[index] = b_j;
		dwave_config_b_l[index] = b_l;

		index += 1;
		}
	     }
	  }
	}
    }
 }
	 VectorHost_dble dwave_r_factor_host;
	 dwave_r_factor_host.resizeNoCopy(r_dmn_t::dmn_size());
  	
	 for (int r_i = 0; r_i < r_dmn_t::dmn_size(); r_i++) {
	 dwave_r_factor_host[r_i] = dwave_r_factor(r_i);
	 }


	 VectorHost_dble akima_coefficients_host;
	 MatrixHost_flt G0_sign_up_host;
	 MatrixHost_flt G0_sign_dn_host;

	 MatrixHost_int G0_indices_up_host;
	 MatrixHost_int G0_indices_dn_host;
  	
	 MatrixHost_flt G0_integration_factor_up_host;
	 MatrixHost_flt G0_integration_factor_dn_host;

         akima_nu_nu_r_dmn_t_shifted_t  akm_nunur_dmn_shifted_t;
         akima_coefficients_host.resizeNoCopy(4*nu_nu_r_dmn_t_t_shifted_dmn.get_size());



       index=0;
     ///COPY AKIMA TO HOST MATRIX
                                for (int t_ind = 0; t_ind < shifted_t::dmn_size(); t_ind++){
	    for (int r_ind = 0; r_ind < r_dmn_t::dmn_size(); r_ind++) {
                for (int s1_ind = 0; s1_ind < s::dmn_size(); s1_ind++) {
                    for (int b1_ind = 0; b1_ind < b::dmn_size(); b1_ind++) {
        	        for (int s0_ind = 0; s0_ind < s::dmn_size(); s0_ind++) {
        	            for (int b0_ind = 0; b0_ind < b::dmn_size(); b0_ind++) {
        for (int l = 0; l < 4; l++){
			
			    akima_coefficients_host[index] = akima_coefficients(index);
							index +=1;
							}
				    		}
           	    			}
          	 		}
       	    		}
	 	}
	}

         G0_sign_up_host.resizeNoCopy(std::make_pair(b_r_t_VERTEX_dmn_t::dmn_size(), b_r_t_VERTEX_dmn_t::dmn_size()));
         G0_sign_dn_host.resizeNoCopy(std::make_pair(b_r_t_VERTEX_dmn_t::dmn_size(), b_r_t_VERTEX_dmn_t::dmn_size()));
         
         G0_indices_up_host.resizeNoCopy(std::make_pair(b_r_t_VERTEX_dmn_t::dmn_size(), b_r_t_VERTEX_dmn_t::dmn_size()));
         G0_indices_dn_host.resizeNoCopy(std::make_pair(b_r_t_VERTEX_dmn_t::dmn_size(), b_r_t_VERTEX_dmn_t::dmn_size()));
         
         G0_integration_factor_up_host.resizeNoCopy(std::make_pair(b_r_t_VERTEX_dmn_t::dmn_size(), b_r_t_VERTEX_dmn_t::dmn_size()));
         G0_integration_factor_dn_host.resizeNoCopy(std::make_pair(b_r_t_VERTEX_dmn_t::dmn_size(), b_r_t_VERTEX_dmn_t::dmn_size()));


     /// COPY G0_SIGN, G0_INDICES, G0_INTEGRATION_FACTOR MATRICES TO HOST
	for (int j = 0; j < b_r_t_VERTEX_dmn_t::dmn_size(); j++) {
	    for (int i = 0; i < b_r_t_VERTEX_dmn_t::dmn_size(); i++) {

		 G0_sign_up_host(i,j) = G0_sign_up(i,j);
		 G0_sign_dn_host(i,j) = G0_sign_dn(i,j);
	 
	         G0_indices_up_host(i,j) = G0_indices_up(i,j);
		 G0_indices_dn_host(i,j) = G0_indices_dn(i,j);

	 	 G0_integration_factor_up_host(i,j) = G0_integration_factor_up(i,j);
 		 G0_integration_factor_dn_host(i,j) = G0_integration_factor_dn(i,j);
		 
	}
    }



  std::vector<std::vector<double>> value_r;
  int rcluster_size = RClusterDmn::parameter_type::get_size();

  VectorHost_dble r_abs_diff;
  r_abs_diff.resizeNoCopy(rcluster_size);

//  std::cout<<"size of domain: "<<rcluster_size<<std::endl;
  for (int a=0; a<rcluster_size; a++) 
  {
  value_r.push_back(RClusterDmn::parameter_type::get_elements()[a]);
//  std::cout<<value_r[a][0]<<" "<<value_r[a][1]<<std::endl;
  r_abs_diff[a] = sqrt(value_r[a][0]*value_r[a][0] + value_r[a][1]*value_r[a][1]);
  }


    const auto& sub_mat_r = RClusterDmn::parameter_type::get_subtract_matrix();
    const static double beta = beta_;
    const static double N_div_beta = sp_time_intervals_/ beta;

    TpEqTimeHelper::set( sub_mat_r.ptr(),sub_mat_r.leadingDimension(), RClusterDmn::dmn_size(), G0_indices_up_host.ptr(), G0_indices_up_host.leadingDimension(),
 			G0_indices_dn_host.ptr(), G0_indices_dn_host.leadingDimension(),
                        G0_sign_up_host.ptr(), G0_sign_up_host.leadingDimension(), G0_sign_dn_host.ptr(),G0_sign_dn_host.leadingDimension(), 
                        G0_integration_factor_up_host.ptr(), G0_integration_factor_up_host.leadingDimension(),
			G0_integration_factor_dn_host.ptr(), G0_integration_factor_dn_host.leadingDimension(),
			G0_original_up.ptr(), G0_original_up.leadingDimension(), G0_original_dn.ptr(), G0_original_dn.leadingDimension(),
                        b_r_t_VERTEX_dmn_t::dmn_size(),t_VERTEX::dmn_size(),
 			akima_coefficients_host.ptr(), 4, b::dmn_size(), s::dmn_size(), r_dmn_t::dmn_size(), shifted_t::dmn_size(),akima_coefficients.size(),
			fixed_config_b_ind.ptr(), fixed_config_r_ind.ptr(), fixed_config_t_ind.ptr(), fixed_config_t_val.ptr(),
			r_abs_diff.ptr(),rcluster_size,
  			beta, N_div_beta,dwave_config_r_i.ptr(),dwave_config_r_j.ptr(),dwave_config_r_l.ptr(),dwave_config_b_i.ptr(),dwave_config_b_j.ptr(),dwave_config_b_l.ptr(),dwave_config_size, dwave_r_factor_host.ptr());
    assert(cudaPeekAtLastError() == cudaSuccess);

//         std::cout<<"Copied data to GPU successfullly........"<<std::endl;


   fixed_config_b_ind.clear();
   fixed_config_r_ind.clear();
   fixed_config_t_ind.clear();
   fixed_config_t_val.clear();

   akima_coefficients_host.clear();

   G0_sign_up_host.clear();
   G0_sign_dn_host.clear();
   G0_indices_up_host.clear();
   G0_indices_dn_host.clear();

  G0_integration_factor_up_host.clear();
  G0_integration_factor_dn_host.clear();

   }); //end of call_once

}


template <class parameters_type, class MOMS_type>
void TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::GPU>::resetAccumulation() {
  GFLOP = 0;

  G_r_t_accumulated_dev.setToZeroAsync(streams_[0]);
  assert(cudaPeekAtLastError() == cudaSuccess);

  G_r_t_accumulated_squared_dev.setToZeroAsync(streams_[1]);
  assert(cudaPeekAtLastError() == cudaSuccess);

  spin_ZZ_chi_accumulated_dev.setToZeroAsync(streams_[0]);
  assert(cudaPeekAtLastError() == cudaSuccess);

  spin_ZZ_chi_stddev_dev.setToZeroAsync(streams_[1]);
  assert(cudaPeekAtLastError() == cudaSuccess);

  spin_XX_chi_accumulated_dev.setToZeroAsync(streams_[0]);
  assert(cudaPeekAtLastError() == cudaSuccess);

  dwave_pp_correlator_dev.setToZeroAsync(streams_[1]);
  assert(cudaPeekAtLastError() == cudaSuccess);



}

template <class parameters_type, class MOMS_type>
void TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::GPU>::interpolate(
    func::function<double, func::dmn_variadic<nu, nu, r_dmn_t, t>>& G_r_t,
    func::function<double, func::dmn_variadic<nu, nu, r_dmn_t, t>>& G_r_t_stddev) {
  int size = t_VERTEX::dmn_size();

  math::interpolation::akima_interpolation<double> ai_obj(size);

  double* x = new double[size];
  double* y = new double[size];

  for (int t_ind = 0; t_ind < t_VERTEX::dmn_size(); t_ind++)
    x[t_ind] = t_VERTEX::get_elements()[t_ind];

  {
    for (int r_ind = 0; r_ind < r_dmn_t::dmn_size(); r_ind++) {
      for (int nu1_ind = 0; nu1_ind < nu::dmn_size(); nu1_ind++) {
        for (int nu0_ind = 0; nu0_ind < nu::dmn_size(); nu0_ind++) {
          {
            for (int t_ind = 0; t_ind < t_VERTEX::dmn_size(); t_ind++)
              y[t_ind] = G_r_t_accumulated(nu0_ind, nu1_ind, r_ind, t_ind);

            ai_obj.initialize(x, y);

            for (int t_ind = t::dmn_size() / 2; t_ind < t::dmn_size(); t_ind++)
              G_r_t(nu0_ind, nu1_ind, r_ind, t_ind) = ai_obj.evaluate(t::get_elements()[t_ind]);
          }

          {
            for (int t_ind = 0; t_ind < t_VERTEX::dmn_size(); t_ind++)
              y[t_ind] = G_r_t_accumulated_squared(nu0_ind, nu1_ind, r_ind, t_ind);

            ai_obj.initialize(x, y);

            for (int t_ind = t::dmn_size() / 2; t_ind < t::dmn_size(); t_ind++)
              G_r_t_stddev(nu0_ind, nu1_ind, r_ind, t_ind) =
                  ai_obj.evaluate(t::get_elements()[t_ind]);
          }
        }
      }
    }
    for (int r_ind = 0; r_ind < r_dmn_t::dmn_size(); r_ind++)
      for (int nu1_ind = 0; nu1_ind < nu::dmn_size(); nu1_ind++)
        for (int nu0_ind = 0; nu0_ind < nu::dmn_size(); nu0_ind++)
          for (int t_ind = 0; t_ind < t::dmn_size() / 2; t_ind++)
            G_r_t(nu0_ind, nu1_ind, r_ind, t_ind) =
                -G_r_t(nu0_ind, nu1_ind, r_ind, t_ind + t::dmn_size() / 2);

    for (int r_ind = 0; r_ind < r_dmn_t::dmn_size(); r_ind++)
      for (int nu1_ind = 0; nu1_ind < nu::dmn_size(); nu1_ind++)
        for (int nu0_ind = 0; nu0_ind < nu::dmn_size(); nu0_ind++)
          for (int t_ind = 0; t_ind < t::dmn_size() / 2; t_ind++)
            G_r_t_stddev(nu0_ind, nu1_ind, r_ind, t_ind) =
                G_r_t_stddev(nu0_ind, nu1_ind, r_ind, t_ind + t::dmn_size() / 2);
  }

  delete[] x;
  delete[] y;
}

template <class parameters_type, class MOMS_type>
void TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::GPU>::finalize() {

  synchronizeStreams();


  cudaMemcpy(G_r_t_accumulated.values(), G_r_t_accumulated_dev.ptr(), G_r_t_accumulated.size()* sizeof(double), 
             cudaMemcpyDeviceToHost);

  cudaMemcpy(G_r_t_accumulated_squared.values(), G_r_t_accumulated_squared_dev.ptr(), G_r_t_accumulated_squared.size()* sizeof(double), 
             cudaMemcpyDeviceToHost);

  cudaMemcpy(spin_ZZ_chi_accumulated.values(), spin_ZZ_chi_accumulated_dev.ptr(), spin_ZZ_chi_accumulated.size()* sizeof(double), 
             cudaMemcpyDeviceToHost);

  cudaMemcpy(spin_ZZ_chi_stddev.values(), spin_ZZ_chi_stddev_dev.ptr(), spin_ZZ_chi_stddev.size()* sizeof(double), 
             cudaMemcpyDeviceToHost);

  cudaMemcpy(spin_XX_chi_accumulated.values(), spin_XX_chi_accumulated_dev.ptr(), spin_XX_chi_accumulated.size()* sizeof(double), 
             cudaMemcpyDeviceToHost);

  cudaMemcpy(dwave_pp_correlator.values(), dwave_pp_correlator_dev.ptr(), dwave_pp_correlator.size()* sizeof(double), 
             cudaMemcpyDeviceToHost);


  synchronizeStreams();

  for (int l = 0; l < G_r_t_accumulated_squared.size(); l++)
    G_r_t_accumulated_squared(l) =
        std::sqrt(std::abs(G_r_t_accumulated_squared(l) - std::pow(G_r_t_accumulated(l), 2)));

   interpolate(G_r_t, G_r_t_stddev);

}


template <class parameters_type, class MOMS_type>
template <class configuration_type, typename RealInp>
void TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::GPU>::compute_G_r_t(
    const configuration_type& configuration_e_up,
    const dca::linalg::Matrix<RealInp, linalg::GPU>& M_up,
    const configuration_type& configuration_e_dn,
    const dca::linalg::Matrix<RealInp, linalg::GPU>& M_dn) {


  int config_size_up = find_first_non_interacting_spin(configuration_e_up);
  int config_size_dn = find_first_non_interacting_spin(configuration_e_dn);
  config_up_.resize(config_size_up);
  config_dn_.resize(config_size_dn);

  for (int i = 0; i < config_size_up; ++i) {
  	config_up_[i].band = configuration_e_up[i].get_band();
  	config_up_[i].rsite = configuration_e_up[i].get_r_site();
  	config_up_[i].tau = configuration_e_up[i].get_tau();
  }

  for (int i = 0; i < config_size_dn; ++i) {
  	config_dn_[i].band = configuration_e_dn[i].get_band();
  	config_dn_[i].rsite = configuration_e_dn[i].get_r_site();
  	config_dn_[i].tau = configuration_e_dn[i].get_tau();
  }


  config_up_dev_.setAsync(config_up_, streams_[1]);
  config_dn_dev_.setAsync(config_dn_, streams_[0]);
  

  int spin_index_up = 1;
  int spin_index_dn = 0;

  M_matrix_dn_dev.resizeNoCopy(std::pair<int, int>(config_size_dn, config_size_dn));
  G0_matrix_dn_left_dev.resizeNoCopy(
        std::pair<int, int>(b_r_t_VERTEX_dmn_t::dmn_size(), config_size_dn));
  G0_matrix_dn_right_dev.resizeNoCopy(
        std::pair<int, int>(config_size_dn, b_r_t_VERTEX_dmn_t::dmn_size()));

  M_G0_matrix_dn_dev.resizeNoCopy(
        std::pair<int, int>(config_size_dn, b_r_t_VERTEX_dmn_t::dmn_size()));


  M_matrix_up_dev.resizeNoCopy(std::pair<int, int>(config_size_up, config_size_up));

  G0_matrix_up_left_dev.resizeNoCopy(
        std::pair<int, int>(b_r_t_VERTEX_dmn_t::dmn_size(), config_size_up));
  G0_matrix_up_right_dev.resizeNoCopy(
        std::pair<int, int>(config_size_up, b_r_t_VERTEX_dmn_t::dmn_size()));

  M_G0_matrix_up_dev.resizeNoCopy(
        std::pair<int, int>(config_size_up, b_r_t_VERTEX_dmn_t::dmn_size()));
         
  G0_M_G0_matrix_dn_dev.resizeNoCopy(
      				std::pair<int, int>(b_r_t_VERTEX_dmn_t::dmn_size(), b_r_t_VERTEX_dmn_t::dmn_size()));
  G0_M_G0_matrix_up_dev.resizeNoCopy(
      				std::pair<int, int>(b_r_t_VERTEX_dmn_t::dmn_size(), b_r_t_VERTEX_dmn_t::dmn_size()));
  



  synchronizeStreams();

  calc_G_r_t_OnDevice(spin_index_up, M_up.ptr(), M_up.leadingDimension(), M_matrix_up_dev.ptr(), M_matrix_up_dev.leadingDimension(),
                      G0_matrix_up_left_dev.ptr(), G0_matrix_up_left_dev.leadingDimension(), G0_matrix_up_right_dev.ptr(), G0_matrix_up_right_dev.leadingDimension(),
		      M_G0_matrix_up_dev.ptr(), M_G0_matrix_up_dev.leadingDimension(), G0_M_G0_matrix_up_dev.ptr(), G0_M_G0_matrix_up_dev.leadingDimension(),
		      config_up_dev_.ptr(), config_size_up, G_r_t_up_dev.ptr(), G_r_t_up_dev.leadingDimension(), b_r_t_VERTEX_dmn_t::dmn_size(), 
		      streams_[0], streams_id_[0], thread_id);

  assert(cudaPeekAtLastError() == cudaSuccess);


  calc_G_r_t_OnDevice(spin_index_dn, M_dn.ptr(), M_dn.leadingDimension(), M_matrix_dn_dev.ptr(), M_matrix_dn_dev.leadingDimension(),
                      G0_matrix_dn_left_dev.ptr(), G0_matrix_dn_left_dev.leadingDimension(), G0_matrix_dn_right_dev.ptr(), G0_matrix_dn_right_dev.leadingDimension(),
		      M_G0_matrix_dn_dev.ptr(), M_G0_matrix_dn_dev.leadingDimension(), G0_M_G0_matrix_dn_dev.ptr(), G0_M_G0_matrix_dn_dev.leadingDimension(),
		      config_dn_dev_.ptr(), config_size_dn, G_r_t_dn_dev.ptr(), G_r_t_dn_dev.leadingDimension(), b_r_t_VERTEX_dmn_t::dmn_size(), 
		      streams_[1], streams_id_[1], thread_id);

  assert(cudaPeekAtLastError() == cudaSuccess);






}

template <class parameters_type, class MOMS_type>
void TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::GPU>::accumulate_G_r_t(double sign) {
  for (int j = 0; j < b_r_t_VERTEX_dmn_t::dmn_size(); j++) {
    for (int i = 0; i < b_r_t_VERTEX_dmn_t::dmn_size(); i++) {
      G_r_t_accumulated(G0_indices_dn(i, j)) +=
          sign * G0_integration_factor_dn(i, j) * G_r_t_dn(i, j);
      G_r_t_accumulated_squared(G0_indices_dn(i, j)) +=
          sign * G0_integration_factor_dn(i, j) * G_r_t_dn(i, j) * G_r_t_dn(i, j);

      G_r_t_accumulated(G0_indices_up(i, j)) +=
          sign * G0_integration_factor_up(i, j) * G_r_t_up(i, j);
      G_r_t_accumulated_squared(G0_indices_up(i, j)) +=
          sign * G0_integration_factor_up(i, j) * G_r_t_up(i, j) * G_r_t_up(i, j);
    }
  }
}


template <class parameters_type, class MOMS_type>
int TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::GPU>::find_first_non_interacting_spin(
    const std::vector<vertex_singleton_type>& configuration_e_spin) {
  int configuration_size = configuration_e_spin.size();

  int vertex_index = 0;
  while (vertex_index < configuration_size &&
         configuration_e_spin[vertex_index].get_HS_spin() != HS_ZERO)
    vertex_index++;

  assert(vertex_index == configuration_size ||
         configuration_e_spin[vertex_index].get_HS_spin() == HS_ZERO);

  return vertex_index;
}

template <class parameters_type, class MOMS_type>
template <class configuration_type, typename RealInp>
void TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::GPU>::accumulateAll(
    const configuration_type& configuration_e_up,
    const dca::linalg::Matrix<RealInp, dca::linalg::CPU>& M_up_host,
    const configuration_type& configuration_e_dn,
    const dca::linalg::Matrix<RealInp, dca::linalg::CPU>& M_dn_host, int sign) {
  

    dca::linalg::Matrix<RealInp, linalg::GPU> M_up_dev;
    dca::linalg::Matrix<RealInp, linalg::GPU> M_dn_dev;

    M_up_dev.setAsync(M_up_host, streams_[0]);
    M_dn_dev.setAsync(M_dn_host, streams_[1]);

    synchronizeStreams();
    accumulateAll(configuration_e_up, M_up_dev, configuration_e_dn, M_dn_dev, sign);
}

template <class parameters_type, class MOMS_type>
template <class configuration_type, typename RealInp>
void TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::GPU>::accumulateAll(
    const configuration_type& configuration_e_up,
    const dca::linalg::Matrix<RealInp, dca::linalg::GPU>& M_up,
    const configuration_type& configuration_e_dn,
    const dca::linalg::Matrix<RealInp, dca::linalg::GPU>& M_dn, int sign) {



  initialize_TpEqTime_helper();

  synchronizeStreams();

  compute_G_r_t(configuration_e_up, M_up, configuration_e_dn, M_dn);
  assert(cudaPeekAtLastError() == cudaSuccess);

  synchronizeStreams();

  

  accumulate_G_r_t_OnDevice(G_r_t_up_dev.ptr(), G_r_t_up_dev.leadingDimension(), G_r_t_dn_dev.ptr(), G_r_t_dn_dev.leadingDimension(), static_cast<RealInp>(sign), G_r_t_accumulated_dev.ptr(), G_r_t_accumulated_squared_dev.ptr(), b_r_t_VERTEX_dmn_t::dmn_size(), streams_[0] );
  assert(cudaPeekAtLastError() == cudaSuccess);

  accumulate_chi_OnDevice(G_r_t_up_dev.ptr(), G_r_t_up_dev.leadingDimension(), G_r_t_dn_dev.ptr(), G_r_t_dn_dev.leadingDimension(), static_cast<RealInp>(sign) ,spin_ZZ_chi_accumulated_dev.ptr(),  spin_ZZ_chi_stddev_dev.ptr(), spin_XX_chi_accumulated_dev.ptr(), b_r_t_VERTEX_dmn_t::dmn_size(), r_dmn_t::dmn_size() ,t_VERTEX::dmn_size(), streams_[1]);
  assert(cudaPeekAtLastError() == cudaSuccess);

  accumulate_dwave_pp_correlator_OnDevice(G_r_t_up_dev.ptr(), G_r_t_up_dev.leadingDimension(), G_r_t_dn_dev.ptr(), G_r_t_dn_dev.leadingDimension(), static_cast<RealInp>(sign) ,dwave_pp_correlator_dev.ptr(), t_VERTEX::dmn_size(), r_dmn_t::dmn_size(), dwave_config_size, streams_[0]);
  assert(cudaPeekAtLastError() == cudaSuccess);
//  accumulate_moments(sign);

//  accumulate_dwave_pp_correlator(sign);



  synchronizeStreams();

  event_.record(streams_[0]);
  event_.record(streams_[1]);

}

template <class parameters_type, class MOMS_type>
void TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::GPU>::accumulate_G_r_t_orig(double sign) {
  for (int j = 0; j < b_r_t_VERTEX_dmn_t::dmn_size(); j++) {
    for (int i = 0; i < b_r_t_VERTEX_dmn_t::dmn_size(); i++) {
      G_r_t_accumulated(G0_indices_dn(i, j)) +=
          sign * G0_integration_factor_dn(i, j) * G_r_t_dn(i, j);
      G_r_t_accumulated_squared(G0_indices_dn(i, j)) +=
          sign * G0_integration_factor_dn(i, j) * G_r_t_dn(i, j) * G_r_t_dn(i, j);

      G_r_t_accumulated(G0_indices_up(i, j)) +=
          sign * G0_integration_factor_up(i, j) * G_r_t_up(i, j);
      G_r_t_accumulated_squared(G0_indices_up(i, j)) +=
          sign * G0_integration_factor_up(i, j) * G_r_t_up(i, j) * G_r_t_up(i, j);
    }
  }
}
template <class parameters_type, class MOMS_type>
void TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::GPU>::synchronizeStreams() {
  for (auto stream : streams_)
    cudaStreamSynchronize(stream);
}



template <class parameters_type, class MOMS_type>
void TpEqualTimeAccumulator<parameters_type, MOMS_type, linalg::GPU>::sumTo(this_type& other) {

  
  cudaStreamSynchronize(other.streams_[0]);
  cudaStreamSynchronize(other.streams_[1]);

  sum_OnDevice(G_r_t_accumulated_dev.ptr(), other.G_r_t_accumulated_dev.ptr(), G_r_t_accumulated_dev.size(),streams_[0]);
  sum_OnDevice(G_r_t_accumulated_squared_dev.ptr(), other.G_r_t_accumulated_squared_dev.ptr(), G_r_t_accumulated_squared_dev.size(),streams_[1]);
  sum_OnDevice(spin_ZZ_chi_accumulated_dev.ptr(), other.spin_ZZ_chi_accumulated_dev.ptr(), spin_ZZ_chi_accumulated_dev.size(),streams_[0]);
  sum_OnDevice(spin_ZZ_chi_stddev_dev.ptr(), other.spin_ZZ_chi_stddev_dev.ptr(), spin_ZZ_chi_stddev_dev.size(),streams_[1]);
  sum_OnDevice(spin_XX_chi_accumulated_dev.ptr(), other.spin_XX_chi_accumulated_dev.ptr(), spin_XX_chi_accumulated_dev.size(),streams_[0]);
  sum_OnDevice(dwave_pp_correlator_dev.ptr(), other.dwave_pp_correlator_dev.ptr(), dwave_pp_correlator_dev.size(),streams_[1]);

  synchronizeStreams();

  return;

}

}  // namespace ctaux
}  // namespace solver
}  // namespace phys
}  // namespace dca

#endif  // DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_CTAUX_ACCUMULATOR_TP_TP_EQUAL_TIME_ACCUMULATOR_HPP
